import threading
import time
from collections import deque
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

from aibridge.llm import LLM  # or wherever your LLM base class resides

class LoadBalancedLLM(LLM):
    """
    Wraps another LLM and enforces:
      - A rate limit on requests_per_minute.
      - Limited concurrency (via a ThreadPoolExecutor).
    Accumulates combined token usage/cost only if the wrapped LLM supports it.
    """

    def __init__(self, llm: LLM, max_requests_per_minute: int, max_concurrent_requests: int):
        """
        :param llm:                     The underlying LLM to wrap.
        :param max_requests_per_minute: Maximum number of requests per minute allowed.
        :param max_concurrent_requests: Maximum number of parallel requests to process.
        """
        # Check if the child LLM has numerical cost fields
        child_has_cost = (
            llm.cost_per_1M_tokens_input is not None
            and llm.cost_per_1M_tokens_output is not None
        )

        # Rebuild a cost structure dict if the child has cost fields
        if child_has_cost:
            cost_dict = {
                "cost_per_1M_tokens_input": llm.cost_per_1M_tokens_input,
                "cost_per_1M_tokens_output": llm.cost_per_1M_tokens_output
            }
            super().__init__(cost_dict)
        else:
            super().__init__(None)

        # Store the wrapped LLM
        self.llm = llm
        # Keep a flag to know if child LLM tracks cost
        self._child_has_cost = child_has_cost

        # Rate-limiting and concurrency config
        self.requests_per_minute = max_requests_per_minute
        self.concurrency = max_concurrent_requests

        # Internal structures for rate limiting
        self.timestamps = deque()  # Tracks request timestamps (for rate-limiting)
        self.queue = Queue()       # Thread-safe queue for requests
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        # Thread pool for concurrency control
        self.executor = ThreadPoolExecutor(max_workers=self.concurrency)

        # Start worker thread to monitor the queue and dispatch tasks
        self.worker_thread = threading.Thread(target=self._queue_loop, daemon=True)
        self.worker_thread.start()

    def get_completion(self, prompt):
        """
        Enqueues a request to get a completion from the wrapped LLM.
        Blocks until the request is done, then returns the text result.
        """
        event = threading.Event()  # Signaled once the request finishes
        result_holder = {}

        with self.lock:
            # Put our request into the queue with the current time
            request_time = time.time()
            self.queue.put((request_time, prompt, event, result_holder))

        # Wait until the background worker signals completion
        event.wait()
        return result_holder["completion"]

    def _process_request(self, prompt, event, result_holder):
        """
        Actual request processing: calls the underlying LLM and (optionally) updates cost usage.
        """
        try:
            # If the child has cost-tracking, measure usage before call
            if self._child_has_cost:
                usage_before = self.llm.get_token_counter()

            # Call the wrapped LLM to get a completion
            completion = self.llm.get_completion(prompt)

            # If the child has cost-tracking, measure usage after call
            if self._child_has_cost:
                usage_after = self.llm.get_token_counter()

                # Determine how many tokens were used for this request
                used_input = usage_after["input"] - usage_before["input"]
                used_output = usage_after["output"] - usage_before["output"]

                # Update this wrapper's counters
                self.update_token_counters(used_input, used_output)

            # Store the result
            result_holder["completion"] = completion

        finally:
            # Signal that we're done with this request
            event.set()

    def _queue_loop(self):
        """
        Continuously checks the queue and processes requests,
        respecting the requests_per_minute limit.
        """
        while not self.stop_event.is_set():
            now = time.time()

            # Remove timestamps older than 60 seconds for rate-limiting
            while self.timestamps and self.timestamps[0] < now - 60:
                self.timestamps.popleft()

            # If there's capacity for a new request
            if not self.queue.empty() and len(self.timestamps) < self.requests_per_minute:
                # Fetch next request
                request_time, prompt, event, result_holder = self.queue.get()

                # Record the timestamp for rate-limiting
                self.timestamps.append(request_time)

                # Dispatch processing to the thread pool
                self.executor.submit(self._process_request, prompt, event, result_holder)

            # Small sleep to avoid busy-waiting
            time.sleep(0.05)

    def stop_worker(self):
        """
        Cleanly stop the background worker and shut down the executor.
        """
        self.stop_event.set()
        self.worker_thread.join()
        self.executor.shutdown(wait=True)

    def identify(self):
        """
        For clarity, identify as LoadBalancedLLM wrapping the underlying LLM's identify().
        """
        return f"LoadBalancedLLM({self.llm.identify()})"

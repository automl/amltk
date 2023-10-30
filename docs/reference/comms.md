## Comm Plugin
!!! todo "Update these docs"

    Sorry

The [`Comm`][amltk.scheduling.Comm] is the access point to to enable
two way communication between a worker and the server.

??? warning "Local Processes Only"

    We currently use [`multiprocessing.Pipe`][multiprocessing.Pipe] to communicate
    between the worker and the scheduler. This means we are limited to local processes
    only.

    If there is interest, we could extend this to be interfaced and provide web socket
    communication as well. Please open an issue if you are interested in this or if you
    would like to contribute.

### Usage of Comm Plugin
A [`Comm`][amltk.Comm] facilitate the communication between the worker and the scheduler.
By using this `Comm`, we can [`send()`][amltk.Comm.send] and
[`request()`][amltk.Comm.request] messages from the workers point of view.
These messages are then received by the scheduler and emitted as the
[`MESSAGE`][amltk.Comm.MESSAGE] and [`REQUEST`][amltk.Comm.REQUEST]
events respectively which both pass a [`Comm.Msg`][amltk.Comm.Msg] object
to the callback. This object contains the `data` that was transmitted.

Below we show an example of both `send()` and
`request()` in action and how to use the plugin.

=== "`send()`"

    ```python hl_lines="7 9 12 16 17 18 19 20 21"
    from amltk import Scheduler, Comm

    # The function must accept an optional `Comm` keyword argument
    def echoer(xs: list[int], comm: Comm | None = None):
        assert comm is not None

        with comm:  # (1)!
          for x in xs:
              comm.send(x)  # (2)!

    scheduler = Scheduler.with_processes(1)
    task = scheduler.task(echoer, plugin=Comm.Plugin())

    @scheduler.on_start
    def start():
        task.submit([1, 2, 3, 4, 5])

    @task.on(Comm.MESSAGE)
    def on_message(msg: Comm.Msg):  # (3)!
        print(f"Recieved a message {msg=}")
        print(msg.data)

    scheduler.run()
    ```

    1. The `Comm` object should be used as a context manager. This is to ensure
       that the `Comm` object is closed correctly when the function exits.
    2. Here we use the [`send()`][amltk.Comm.send] method to send a message
       to the scheduler.
    3. We can also do `#!python Comm.Msg[int]` to specify the type of data
       we expect to receive.

=== "`request()`"

    ```python hl_lines="7 16 17 18 19"
    from amltk.scheduling import Scheduler, Comm

    # The function must accept an optional `Comm` keyword argument
    def requester(xs: list[int], comm: Comm | None = None):
        with comm:
          for _ in range(n):
              response = comm.request(n)  # (1)!

    scheduler = Scheduler(...)
    task = scheduler.task(requester, plugin=Comm.Plugin())

    @scheduler.on_start
    def start():
        task.submit([1, 2, 3, 4, 5])

    @task.on_request
    def handle_request(msg: Comm.Msg):
        print(f"Recieved request {msg=}")
        msg.respond(msg.data * 2)  # (2)!

    scheduler.run()
    ```

    1. Here we use the [`request()`][amltk.Comm.request] method to send a request
       to the scheduler with some data.
    2. We can use the [`respond()`][amltk.Comm.Msg.respond] method to
       respond to the request with some data.

!!! tip "Identifying Workers"

    The [`Comm.Msg`][amltk.Comm.Msg] object also has the `identifier`
    attribute, which is a unique identifier for the worker.

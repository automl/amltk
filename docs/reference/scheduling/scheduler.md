## Scheduler
The [`Scheduler`][amltk.scheduling.Scheduler] uses
an [`Executor`][concurrent.futures.Executor], a builtin python native with
a `#!python submit(f, *args, **kwargs)` function to submit compute to
be compute else where, whether it be locally or remotely.

The `Scheduler` is primarily used to dispatch compute to an `Executor` and
emit `@events`, which can trigger user callbacks.

Typically you should not use the `Scheduler` directly for dispatching and
responding to computed functions, but rather use a [`Task`][amltk.scheduling.Task]

??? note "Running in a Jupyter Notebook/Colab"

    If you are using a Jupyter Notebook, you likley need to use the following
    at the top of your notebook:

    ```python
    import nest_asyncio  # Only necessary in Notebooks
    nest_asyncio.apply()

    scheduler.run(...)
    ```

    This is due to the fact a notebook runs in an async context. If you do not
    wish to use the above snippet, you can instead use:

    ```python
    await scheduler.async_run(...)
    ```

??? tip "Basic Usage"

    In this example, we create a scheduler that uses local processes as
    workers. We then create a task that will run a function `fn` and submit it
    to the scheduler. Lastly, a callback is registered to `@future-result` to print the
    result when the compute is done.

    ```python exec="true" source="material-block" html="true"
    from amltk.scheduling import Scheduler

    def fn(x: int) -> int:
        return x + 1
    from amltk._doc import make_picklable; make_picklable(fn)  # markdown-exec: hide

    scheduler = Scheduler.with_processes(1)

    @scheduler.on_start
    def launch_the_compute():
        scheduler.submit(fn, 1)

    @scheduler.on_future_result
    def callback(future, result):
        print(f"Result: {result}")

    scheduler.run()
    from amltk._doc import doc_print; doc_print(print, scheduler)  # markdown-exec: hide
    ```

    The last line in the previous example called
    [`scheduler.run()`][amltk.scheduling.Scheduler.run] is what starts the scheduler
    running, in which it will first emit the `@start` event. This triggered the
    callback `launch_the_compute()` which submitted the function `fn` with the
    arguments `#!python 1`.

    The scheduler then ran the compute and waited for it to complete, emitting the
    `@future-result` event when it was done successfully. This triggered the callback
    `callback()` which printed the result.

    At this point, there is no more compute happening and no more events to respond to
    so the scheduler will halt.

??? example "`@events`"

    === "Scheduler Status Events"

        When the scheduler enters some important state, it will emit an event
        to let you know.

        === "`@start`"

            ::: amltk.scheduling.Scheduler.on_start
                options:
                    show_root_heading: False
                    show_root_toc_entry: False

        === "`@finishing`"

            ::: amltk.scheduling.Scheduler.on_finishing
                options:
                    show_root_heading: False
                    show_root_toc_entry: False

        === "`@finished`"

            ::: amltk.scheduling.Scheduler.on_finished
                options:
                    show_root_heading: False
                    show_root_toc_entry: False

        === "`@stop`"

            ::: amltk.scheduling.Scheduler.on_stop
                options:
                    show_root_heading: False
                    show_root_toc_entry: False

        === "`@timeout`"

            ::: amltk.scheduling.Scheduler.on_timeout
                options:
                    show_root_heading: False
                    show_root_toc_entry: False

        === "`@empty`"

            ::: amltk.scheduling.Scheduler.on_empty
                options:
                    show_root_heading: False
                    show_root_toc_entry: False

    === "Submitted Compute Events"

        When any compute goes through the `Scheduler`, it will emit an event
        to let you know. You should however prefer to use a
        [`Task`][amltk.scheduling.Task] as it will emit specific events
        for the task at hand, and not all compute.

        === "`@future-submitted`"

            ::: amltk.scheduling.Scheduler.on_future_submitted
                options:
                    show_root_heading: False
                    show_root_toc_entry: False

        === "`@future-result`"

            ::: amltk.scheduling.Scheduler.on_future_result
                options:
                    show_root_heading: False
                    show_root_toc_entry: False

        === "`@future-exception`"

            ::: amltk.scheduling.Scheduler.on_future_exception
                options:
                    show_root_heading: False
                    show_root_toc_entry: False

        === "`@future-done`"

            ::: amltk.scheduling.Scheduler.on_future_done
                options:
                    show_root_heading: False
                    show_root_toc_entry: False

        === "`@future-cancelled`"

            ::: amltk.scheduling.Scheduler.on_future_cancelled
                options:
                    show_root_heading: False
                    show_root_toc_entry: False


??? tip "Common usages of `run()`"

    There are various ways to [`run()`][amltk.scheduling.Scheduler.run] the
    scheduler, notably how long it should run with `timeout=` and also how
    it should react to any exception that may have occurred within the `Scheduler`
    itself or your callbacks.

    Please see the [`run()`][amltk.scheduling.Scheduler.run] API doc for more
    details and features, however we show two common use cases of using the `timeout=`
    parameter.

    You can render a live display using [`run(display=...)`][amltk.scheduling.Scheduler.run].
    This require [`rich`](https://github.com/Textualize/rich) to be installed. You
    can install this with `#!bash pip install rich` or `#!bash pip install amltk[rich]`.


    === "`run(timeout=...)`"

        You can tell the `Scheduler` to stop after a certain amount of time
        with the `timeout=` argument to [`run()`][amltk.scheduling.Scheduler.run].

        This will also trigger the `@timeout` event as seen in the `Scheduler` output.

        ```python exec="true" source="material-block" html="True" hl_lines="19"
        import time
        from asyncio import Future

        from amltk.scheduling import Scheduler

        scheduler = Scheduler.with_processes(1)

        def expensive_function() -> int:
            time.sleep(0.1)
            return 42
        from amltk._doc import make_picklable; make_picklable(expensive_function)  # markdown-exec: hide

        @scheduler.on_start
        def submit_calculations() -> None:
            scheduler.submit(expensive_function)

        # This will endlessly loop the scheduler
        @scheduler.on_future_done
        def submit_again(future: Future) -> None:
            if scheduler.running():
                scheduler.submit(expensive_function)

        scheduler.run(timeout=1)  # End after 1 second
        from amltk._doc import doc_print; doc_print(print, scheduler, output="html", fontsize="small")  # markdown-exec: hide
        ```

    === "`run(timeout=..., wait=False)`"

        By specifying that the `Scheduler` should not wait for ongoing tasks
        to finish, the `Scheduler` will attempt to cancel and possibly terminate
        any running tasks.

        ```python exec="true" source="material-block" html="True"
        import time
        from amltk.scheduling import Scheduler

        scheduler = Scheduler.with_processes(1)

        def expensive_function() -> None:
            time.sleep(10)

        from amltk._doc import make_picklable; make_picklable(expensive_function)  # markdown-exec: hide

        @scheduler.on_start
        def submit_calculations() -> None:
            scheduler.submit(expensive_function)

        scheduler.run(timeout=1, wait=False)  # End after 1 second
        from amltk._doc import doc_print; doc_print(print, scheduler, output="html", fontsize="small")  # markdown-exec: hide
        ```

        ??? info "Forcibly Terminating Workers"

            As an `Executor` does not provide an interface to forcibly
            terminate workers, we provide `Scheduler(terminate=...)` as a custom
            strategy for cleaning up a provided executor. It is not possible
            to terminate running thread based workers, for example using
            `ThreadPoolExecutor` and any Executor using threads to spawn
            tasks will have to wait until all running tasks are finish
            before python can close.

            It's likely `terminate` will trigger the `EXCEPTION` event for
            any tasks that are running during the shutdown, **not***
            a cancelled event. This is because we use a
            [`Future`][concurrent.futures.Future]
            under the hood and these can not be cancelled once running.
            However there is no guarantee of this and is up to how the
            `Executor` handles this.

??? example "Scheduling something to be run later"

    You can schedule some function to be run later using the
    [`#!python scheduler.call_later()`][amltk.scheduling.Scheduler.call_later] method.

    !!! note

        This does not run the function in the background, it just schedules some
        function to be called later, where you could perhaps then use submit to
        scheduler a [`Task`][amltk.scheduling.Task] to run the function in the
        background.

    ```python exec="true" source="material-block" result="python"
    from amltk.scheduling import Scheduler

    scheduler = Scheduler.with_processes(1)

    def fn() -> int:
        print("Ending now!")
        scheduler.stop()

    @scheduler.on_start
    def schedule_fn() -> None:
        scheduler.call_later(1, fn)

    scheduler.run(end_on_empty=False)
    ```

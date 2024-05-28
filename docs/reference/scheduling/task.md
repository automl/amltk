## Tasks
A [`Task`][amltk.scheduling.task.Task] is a unit of work that can be scheduled by the
[`Scheduler`][amltk.scheduling.Scheduler].

It is defined by its `function=` to call. Whenever a `Task`
has its [`submit()`][amltk.scheduling.task.Task.submit] method called,
the function will be dispatched to run by a `Scheduler`.

When a task has returned, either successfully, or with an exception,
it will emit `@events` to indicate so. You can subscribe to these events
with callbacks and act accordingly.


??? example "`@events`"

    Check out the `@events` reference
    for more on how to customize these callbacks. You can also take a look
    at the API of [`on()`][amltk.scheduling.task.Task.on] for more information.

    === "`@on-result`"

        ::: amltk.scheduling.task.Task.on_result
            options:
                show_root_heading: False
                show_root_toc_entry: False

    === "`@on-exception`"

        ::: amltk.scheduling.task.Task.on_exception
            options:
                show_root_heading: False
                show_root_toc_entry: False

    === "`@on-done`"

        ::: amltk.scheduling.task.Task.on_done
            options:
                show_root_heading: False
                show_root_toc_entry: False

    === "`@on-submitted`"

        ::: amltk.scheduling.task.Task.on_submitted
            options:
                show_root_heading: False
                show_root_toc_entry: False

    === "`@on-cancelled`"

        ::: amltk.scheduling.task.Task.on_cancelled
            options:
                show_root_heading: False
                show_root_toc_entry: False

??? tip "Usage"

    The usual way to create a task is with
    [`Scheduler.task()`][amltk.scheduling.scheduler.Scheduler.task],
    where you provide the `function=` to call.

    ```python exec="true" source="material-block" html="true"
    from amltk import Scheduler
    from asyncio import Future

    def f(x: int) -> int:
        return x * 2
    from amltk._doc import make_picklable; make_picklable(f)  # markdown-exec: hide

    scheduler = Scheduler.with_processes(2)
    task = scheduler.task(f)

    @scheduler.on_start
    def on_start():
        task.submit(1)

    @task.on_result
    def on_result(future: Future[int], result: int):
        print(f"Task {future} returned {result}")

    scheduler.run()
    from amltk._doc import doc_print; doc_print(print, scheduler)  # markdown-exec: hide
    ```

    If you'd like to simply just call the original function, without submitting it to
    the scheduler, you can always just call the task directly, i.e. `#!python task(1)`.

You can also provide [`Plugins`][amltk.scheduling.plugins.Plugin] to the task,
to modify tasks, add functionality and add new events.

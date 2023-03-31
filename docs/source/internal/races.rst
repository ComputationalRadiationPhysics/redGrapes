======
Data Races and their Solutions in RedGrapes
======

Task initialization || Task Removal
===================================
If task B is being initialized and requires a dependency to task A,
whose destruction just has been initiated,
it can occur that the in-edge-count of task B's pre-event is increased
but wont ever get decreased because task A's notify_followers() is called before the dependency to task B was created. This leads to a freeze of task-execution since task B will never become ready.

**Solution 1**: remove task from ResourceUserLists before calling notify_followers()?

Does not solve the problem entirely. There is still a possibility for use-after free condition, because the iterator can point at an (at-first) still existing element, rertrieve the pointer to the task which then gets deleted before the dependenc
y is added.

**Solution 2 (current)**: use lock to enforce mutual exclusion of init_graph() and delete_from_resources() *per resource*.

Task Removal
============
A task shall be removed only if it satisfies all following conditions:

* Its post-event is reached, i.e. the function body of the task has finished its execution.
* Its result-get-event is reached, i.e. the value was retrieved from the future object, or the future object was destructed already.

These two conditions appear in both possible orderings, so if a task is ready for removal must be checked whenever one of the two events change its state. These two transitions might be handled on two parallel threads and can therefore create a race condition, where both threads determine to destruct the task resulting in a double-free.

**Solution 1**: use shared_ptr< Task > ?

--> not quite, incompatible with Solution for first Racecondition. Need to know if task is removed in Event::notify(), in order to call delete_from_resources() before notify_followers().
Also std::shared_ptr is prone for cycles.


**Solution 2 (current)**: atomic counter starting from 2. Both the post-event and the result-get-event decrement it. The one which reaches 0 (i.e. the one which is ) deletes the task.


Scheduling || Task Emplacement
==============================
When a new task is created, it shall notify one sleeping worker,
if there is any in order to kickstart the scheduling.
Lets assume all workers are busy.
One of them may be searching for a task and thus not marked free,
so alloc_worker() will not return this worker and wake_one_worker()
will notify no one.
Shortly after that the worker is marked as free and begins to sleep,
but the newly created task will not be executed.
This potentially results in a freeze.

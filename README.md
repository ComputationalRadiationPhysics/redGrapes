# RedGrapes
**Re**source-based, **D**eclarative task-**Gra**phs for **P**arallel, **E**vent-driven **S**cheduling

RedGrapes is a C++14 framework for declaratively creating and scheduling task-graphs, based on high-level resource descriptions.

### Motivation
Writing scalable software using bare threads is hard and error-prone, especially if the workload depends on input parameters and asynchronous operations further complicating the program flow.
For this reason the decoupling of processing stages from their execution is useful because it allows to dynamically schedule them. This is typically done with task-graphs, which are directed acyclic graphs (DAGs), whose vertices are some sort of computation and the edges denote the execution precedence order.
This execution precedence results from the dataflow between the tasks, which gets
complex pretty fast and may also be dynamic which makes it nearly impossible to
manually write explicit task dependencies. So ideally these would be derived
from some sort of high-level description of the dataflow. The goal of this
project is to provide a task-based programming framework, where the task-graph
gets created declaratively.

### Concept
The Program is divided into *tasks*. Those are the smallest unit the scheduler operates with.
Because C++ is mainly a sequential language, the dataflow description is based on *resources* which represent some state shared between tasks. Each task has an annotation how the resources are accessed. Which *access modes* are possible is dependent on the type of the resource. A simple example would be read/write, but also array accesses or composite types could be used. A resource can be associated with a specific *acces mode* forming a *resource access*, of which two can then be used to check if they are conflicting. So each task carries in its *task-properties* a list of
resource-accesses.
If two tasks have conflicting resource-accesses, the first created task is executed
first. So the order of the tasks remains virtually.
When tasks are created, their resource-access-list is compared against the previous
enqueued tasks and corresponding dependencies are created in the task-graph.
The resulting task-graph can then be easily scheduled across parallel threads.

### Example
See [examples](examples) for examples covering more features.

```c++
#include <red_grapes/manager.hpp>
#include <red_grapes/resource/ioresource.hpp>
#include <red_grapes/property/inherit.hpp>
#include <red_grapes/property/resource.hpp>
#include <red_grapes/property/label.hpp>

namespace rg = red_grapes;

using TaskProperties =
    rg::TaskProperties<
        rg::ResourceProperty,
        rg::LabelProperty
    >;

int main()
{
    rg::Manager<TaskProperties, rg::ResourceEnqueuePolicy> mgr( std::thread::hardware_concurrency() );

    rg::IOResource a;

    mgr.emplace_task(
        []{ std::cout << "read from resource a" << std::endl; },
        TaskProperties::Builder()
            .label("Task 1")
            .resources({ a.read() })
    );

    return 0;
}
```

## Documentation
RedGrapes is documented using in-code doxygen comments and reStructured-Text-files (in [docs/source](docs/source)), built with Sphinx.

* [Getting Started](docs/source/tutorial/index.rst)
* [Components](docs/source/components.rst)

## License
This Project is free software, licensed under the [Mozilla MPL 2.0 license](LICENSE).

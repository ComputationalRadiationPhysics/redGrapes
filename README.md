# RedGrapes
**Re**source-based, **D**eclarative task-**Gra**phs for **P**arallel, **E**vent-driven **S**cheduling

[![Language](https://img.shields.io/badge/language-C%2B%2B14-lightgrey)](https://isocpp.org/)
[![License](https://img.shields.io/badge/license-MPL--2.0-blue.svg)](https://www.mozilla.org/en-US/MPL/2.0/)
<hr>

RedGrapes is a C++14 framework for declaratively creating and scheduling task-graphs, based on a high-level resource description.

### Motivation

Modern compute nodes concurrently perform computational tasks over various memory resource pools, cores, and accelerator devices.
In order to achieve high scalability in such a compute node, communication and computation tasks need to be overlapped extensively.

Up until now, software developers that took up to this challenge had to juggle data and in-node execution dependencies manually, which is a tedious and error-prone process.
Real-world workloads that depend on states at runtime and asynchronous communication models complicate the program flow even further.

For this reason, one should decouple aforementioned computational tasks from their execution model altogether.
A typical approach involves task-graphs, which are directed acyclic graphs (DAGs), whose vertices are some sort of computation (or communication) and the edges denote the execution precedence order.
The execution precedence arises from the order in which those tasks were declared by the programmer but also have to take into account the data dependencies between the tasks (which are also a DAG).

Consequently, **RedGrapes** provides you with a light-weight, application-level, task-based C++ programming framework.
Herein, a task-graph is generated declaratively from access to resources and order of your code, just as in serial programming.

### Concept

The program shall be divided into **tasks**.
A task is can be a sub-step in a computational solver, the exchange of data between two memory resource pools, or anything else.
Tasks are the smallest unit the RedGrapes scheduler operates with.
Data dependencies are described via **resources**, which are accessed and potentially manipulated by tasks.

Each task has an annotation how the resources are accessed.
Therein allowed **access modes** depend on the type of the resource.
A simple example would be read/write, but also array accesses or composite types could be used.
A resource can be associated with a specific *access mode* forming a *resource access*, of which two can then be used to check if they are conflicting.
So each task carries in its *task-properties* a list of resource-accesses.
If two tasks have conflicting resource-accesses, the first created task is executed first.
This is exactly the behaviour that one would also achieve when programming serially, without hints given via resources.

When tasks are created, their resource-access list is compared against the previous enqueued tasks and corresponding dependencies are created in the task-graph.
The resulting task-graph is read by a scheduling algorithm that executes individual tasks, e.g. across parallel threads.

### Example

See [examples](examples) for examples covering more features.

```cpp
#include <cassert>
#include <redGrapes/manager.hpp>
#include <redGrapes/resource/ioresource.hpp>
#include <redGrapes/property/resource.hpp>
#include <redGrapes/property/label.hpp>

namespace rg = redGrapes;

using TaskProperties =
    rg::TaskProperties<
        rg::ResourceProperty,
        rg::LabelProperty
    >;

int main()
{
    rg::Manager< TaskProperties, rg::ResourceEnqueuePolicy > mgr;

    rg::IOResource< int > a;
	
    mgr.emplace_task(
        []( auto a ){ *a = 123; },
        TaskProperties::Builder().label("Task 1"),
        a.write()
    );

    mgr.emplace_task(
        []( auto a ){ assert( *a == 123 ); },
        TaskProperties::Builder().label("Task 2"),
        a.read()
    );

    return 0;
}
```

## Documentation

RedGrapes is documented using in-code doxygen comments and reStructured-text files (in [docs/source](docs/source)), build with Sphinx.

* [Getting Started](docs/source/tutorial/index.rst)
* [Components](docs/source/components.rst)

## License

This Project is free software, licensed under the [Mozilla MPL 2.0 license](LICENSE).

## Author Contributions

RedGrapes is developed by members of the [Computational Radiation Physics Group](https://hzdr.de/crp) at [HZDR](https://www.hzdr.de/).
Its conceptual design is based on a [whitepaper by A. Huebl, R. Widera, and M. Matthes (2017)](docs/2017_02_ResourceManagerDraft.pdf). 

* [Michael Sippel](https://github.com/michaelsippel): library design & implementation
* [Dr. Sergei Bastrakov](https://github.com/sbastrakov): supervision
* [Dr. Axel Huebl](https://github.com/ax3l): whitepaper, supervision
* [René Widera](https://github.com/psychocoderHPC): whitepaper, supervision
* [Alexander Matthes](https://github.com/theZiz): whitepaper

### Dependencies

RedGrapes requires a compiler supporting the C++14 standard.
RedGrapes further depends on the following libraries:

* [Boost Graph](https://www.boost.org/doc/libs/1_71_0/libs/graph/doc/)
* [optional for C++14](https://github.com/akrzemi1/Optional) by [Andrzej Krzemienski](https://github.com/akrzemi1)

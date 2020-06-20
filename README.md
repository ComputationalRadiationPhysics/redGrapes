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
The execution precedence arises from the order in which those tasks were declared by the programmer but also have to take into account the data dependencies between the tasks.

Consequently, **RedGrapes** provides you with a light-weight, application-level, task-based C++ programming framework.
Herein, a task-graph is generated declaratively from access to resources and order of your code, just as in serial programming.

### Concept

The program shall be divided into **tasks**.
A task is can be a sub-step in a computational solver, the exchange of data between two memory resource pools, or anything else.
Tasks are the smallest unit the RedGrapes scheduler operates with.
Data dependencies are described via **resources**, which are accessed and potentially manipulated by tasks.

Each task has an annotation how the resources are accessed.
Therein allowed **access modes** depend on the type of the resource.
A simple example would be read/write, but also more complex operations are possible, e.g. on arrays or just a sub view.
A *resource* can be associated with a specific *access mode* forming a **resource access**. These instances of a *resource access* can then be pairwise tested wheter they are conflicting and thereby creating a data-dependency (e.g. two writes to the same resource).
So each task carries a list of these resource-accesses in its so-called **task-properties**.
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

## Comparision with Similar Projects

There are several other libraries and toolchains with similar goals, enabling some kind of task-based programming in C++.
Firstly we should classify such programming systems by how the task-graph is built.
The more low-level approach is to just create tasks as executable unit and **imperatively define task-dependencies**.
This approach may be called "data-driven", because the dependencies can be created by waiting for futures of other tasks. <!--, so basically it is an implementation of an async scheduler.-->
However since we want to achieve **declarative task dependencies**, the aforementioned approach does not suffice and we can exclude this class of runtime-systems.
To enable declarative task dependencies, the runtime must also be aware of shared states and be able to automatically detect conflicting usage in order to derive the task graph.

**compile time checked memory access**: The automatic creation of a task graph is often done via annotations, e.g. a pragma in OpenMP, but that does not guarantee correctness. RedGrapes allows to write relatively safe code in that regard.

**native C++**: PaRSEC has an complicated toolchain using additional compilers, OpenMP makes use of pragmas that require compiler support. RedGrapes only requires the C++14 standard.

**typesafe**: Some libraries like Legion or StarPU use a very old fashioned, untyped argc/argv interface to pass parameters to tasks. This is very dangerous. Both libraries also require in general a lot of C-style boilerplate.

**custom access modes**: RedGrapes supports arbitrary, user-configured access modes, not just read/write, e.g. ranges in arrays, multi-dimensional buffers, etc.

**integration with asynchronous APIs**: To correctly model asynchronous MPI or CUDA calls, the complete operation should be a task, but still not block. The finishing of the asynchronous operation has to be triggered externally. Systems that implement distributed scheduling do not leave this option since the communication is done by the runtime itself.

**inter-process scheduling**: Legion, StarPU, HPX etc. add another layer of abstraction to provide a virtualized programming interface for multiple nodes in a cluster. This implies that the domain decomposition, communication and task-migration is all handled by the runtime. This is out of scope for redGrapes, but could be built on top rather than tightly coupling it.

| **Feature**                                                               | native C++         | typesafe           | custom access modes | compile time checked memory access | CUDA                              | MPI                               | other async APIs               | inter-process scheduling |
|---------------------------------------------------------------------------|--------------------|--------------------|---------------------|------------------------------------|-----------------------------------|-----------------------------------|--------------------------------|--------------------------|
| **declarative task-dependencies**                                         |                    |                    |                     |                                    |                                   |                                   |                                |                          |
| [RedGrapes](https://github.com/ComputationalRadiationPhysics/redGrapes)   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark:                 | :heavy_check_mark: <sup>1,2</sup> | :heavy_check_mark: <sup>1,2</sup> | :heavy_check_mark:<sup>3</sup> | :x:                      |
| [SYCL](https://www.khronos.org/sycl/)                                     | :heavy_check_mark: | :heavy_check_mark: | :x:                 | :heavy_check_mark:                 | :x:                               | :x:                               | :x: <sup>4</sup>               | :x:                      |
| [MetaPass](http://www.jlifflander.com/papers/meta-espm2016.pdf)           | :heavy_check_mark: | :heavy_check_mark: | :x:                 | :heavy_check_mark:                 | :heavy_check_mark: <sup>5</sup>   | :heavy_check_mark: <sup>5</sup>   | :x:                            | :heavy_check_mark:       |
| [Legion](https://legion.stanford.edu/)                                    | :heavy_check_mark: | :x:                | :x:                 | :x:                                | :heavy_check_mark: <sup>5</sup>   | :heavy_check_mark: <sup>5</sup>   | :x:                            | :heavy_check_mark:       |
| [StarPU](http://runtime.bordeaux.inria.fr/StarPU/)                        | :heavy_check_mark: | :x:                | :x:                 | :x:                                | :heavy_check_mark: <sup>5</sup>   | :heavy_check_mark: <sup>5</sup>   | :x:                            | :heavy_check_mark:       |
| [PaRSEC](http://icl.cs.utk.edu/parsec/)                                   | :x:                | :heavy_check_mark: | :x:                 | :heavy_check_mark:                 | :heavy_check_mark: <sup>5</sup>   | :heavy_check_mark: <sup>5</sup>   | :x:                            | :heavy_check_mark:       |
| [OpenMP](https://www.openmp.org/)                                         | :x:                | :heavy_check_mark: | :x:                 | :x:                                | :x:                               | :x:                               | :x:                            | :x:                      |
| **imperative task-dependencies**                                          |                    |                    |                     |                                    |                                   |                                   |                                |                          |
| [Realm](http://theory.stanford.edu/~aiken/publications/papers/pact14.pdf) | :heavy_check_mark: | :heavy_check_mark: | -                   | -                                  | :heavy_check_mark: <sup>5</sup>   |                                   | :x:                            | :heavy_check_mark:       |
| [HPX](http://stellar.cct.lsu.edu/projects/hpx/)                           | :heavy_check_mark: | :heavy_check_mark: | -                   | -                                  | :heavy_check_mark: <sup>5</sup>   | :heavy_check_mark: <sup>5</sup>   | :x:                            | :heavy_check_mark:       |
| [TaskFlow](https://taskflow.github.io/)                                   | :heavy_check_mark: | :heavy_check_mark: | -                   | -                                  | :heavy_check_mark: <sup>6</sup>   | :x:                               | :x:                            | :x:                      |

1. user controllable
2. decoupled helper code, but included with redGrapes
3. Events can be triggered extrenally, e.g. from a polling loop
4. See [hipSYCL#181](https://github.com/illuhad/hipSYCL/issues/181)
5. only implicitly managed, not user controlled
6. CUDA-Graphs with TaskFlow-specific wrappers

## License

This Project is free software, licensed under the [Mozilla MPL 2.0 license](LICENSE).

## Author Contributions

RedGrapes is developed by members of the [Computational Radiation Physics Group](https://hzdr.de/crp) at [HZDR](https://www.hzdr.de/).
Its conceptual design is based on a [whitepaper by A. Huebl, R. Widera, and A. Matthes (2017)](docs/2017_02_ResourceManagerDraft.pdf). 

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

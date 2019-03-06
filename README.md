# rmngr
Graph based Resource and Event Management

rmngr is a placeholder name to be found&replaced.

The key-goal of the resource-manager is automatic parallelisation of sequential written programs based on data dependencies.
This happens through annotating functors with their corresponding resource-access information.
The "main program" only calls these functors like they would get executed sequential. With this information (the call order, data dependencies and additional properties) a scheduling-graph is built, which is used to dispatch the jobs to a thread-pool.
This part needs a scheduler to choose which of the ready jobs get dispatched first, but rmngr is not the scheduler itself rather it gives the freedom to implement a scheduler which uses additional information about resources or tasks (e.g. GPU kernels, MPI needs always the same thread,...).

## Documentation
rmngr is documented using in-code doxygen comments and reStructured-Text-files (in [docs/source](docs/source)), built with Sphinx.

* [Getting Started](docs/source/tutorial/index.rst)
* [Components](docs/source/components.rst)

## Licence
TODO



Motivation
==========
Writing scalable software using bare threads is hard and error-prone, especially if the workload depends on input parameters and asynchronous operations further complicating the program flow.
For this reason the decoupling of processing stages from their execution is useful because it allows to dynamically schedule them. This is typically done with task-graphs, which are directed acyclic graphs (DAGs), whose vertices are some sort of computation and the edges denote the execution precedence order.
This execution precedence results from the dataflow between the tasks, which gets
complex pretty fast and may also be dynamic which makes it nearly impossible to
manually write explicit task dependencies. So ideally these would be derived
from some sort of high-level description of the dataflow. The goal of this
project is to provide a task-based programming framework, where the task-graph
gets created declaratively.

Example
=======

TODO



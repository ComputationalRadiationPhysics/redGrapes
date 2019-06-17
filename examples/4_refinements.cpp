
#include <thread>
#include <chrono>

#include <iostream>
#include <rmngr/scheduler/scheduler.hpp>
#include <rmngr/scheduler/resource.hpp>
#include <rmngr/scheduler/dispatch.hpp>
#include <rmngr/scheduler/fifo.hpp>
#include <rmngr/scheduler/graphviz.hpp>

template <typename Graph>
using PrecedenceGraph =
    rmngr::QueuedPrecedenceGraph<
        Graph,
        rmngr::ResourceEnqueuePolicy
    >;

using Scheduler =
    rmngr::Scheduler<
        boost::mpl::vector<
            rmngr::ResourceUserPolicy,
            rmngr::DispatchPolicy< rmngr::FIFO >
        >,
        PrecedenceGraph
    >;

int main( int, char*[] )
{
    Scheduler scheduler(4);

    auto fun1 = scheduler.make_functor(
        [&scheduler]
        {
            std::cout << "f1 on thread " << rmngr::thread::id << "..." << std::endl;

            scheduler.emplace_task(
                []{
                    std::cout << "Refinement 1 on thread " << rmngr::thread::id << std::endl;
                    std::this_thread::sleep_for( std::chrono::seconds(1) );
                });

            scheduler.emplace_task(
                []{
                    std::cout << "Refinement 2 on thread " << rmngr::thread::id << std::endl;
                    std::this_thread::sleep_for( std::chrono::seconds(1) );
                });
        }
    );

    fun1();
    fun1();


    return 0;
}


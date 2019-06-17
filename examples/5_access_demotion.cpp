
#include <thread>
#include <chrono>

#include <iostream>
#include <rmngr/resource/ioresource.hpp>
#include <rmngr/scheduler/scheduler.hpp>
#include <rmngr/scheduler/resource.hpp>
#include <rmngr/scheduler/dispatch.hpp>
#include <rmngr/scheduler/fifo.hpp>

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
    rmngr::IOResource a;

    Scheduler::Properties f1_prop;
    f1_prop.policy<rmngr::ResourceUserPolicy>() += a.write();

    scheduler.emplace_task(
        [&scheduler, a]
        {
            std::cout << "f1 writes A" << std::endl;
            std::this_thread::sleep_for( std::chrono::seconds(1) );

            std::cout << "f1 now only reads A" << std::endl;

            Scheduler::PropertiesPatch patch;
            patch.policy<rmngr::ResourceUserPolicy>() -= a.write();
            patch.policy<rmngr::ResourceUserPolicy>() += a.read();
            scheduler.update_properties(patch);
            std::this_thread::sleep_for( std::chrono::seconds(1) );

            std::cout << "f1 done" << std::endl; 
        },
        f1_prop
    );

    Scheduler::Properties f2_prop;
    f2_prop.policy<rmngr::ResourceUserPolicy>() += a.read();
    scheduler.emplace_task(
        []
        {
            std::cout << "f2 reads A" << std::endl;
            std::cout << "f2 done" << std::endl;
        },
        f2_prop
    );
    
    return 0;
}


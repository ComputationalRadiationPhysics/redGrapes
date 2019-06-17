
#include <iostream>
#include <thread>
#include <chrono>

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

int main(void)
{
    Scheduler scheduler(4);

    rmngr::IOResource a;
    rmngr::IOResource b;

    auto read_a = scheduler.make_functor(
        [=]
        {
            std::cout << "Read from A" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        },
        [=]
        {
            Scheduler::Properties prop;
            prop.policy<rmngr::ResourceUserPolicy>() += a.read();
            return prop;
        }
    );

    auto write_a = scheduler.make_functor(
        [=]
        {
            std::cout << "Write to A" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        },
        [=]
        {
            Scheduler::Properties prop;
            prop.policy<rmngr::ResourceUserPolicy>() += a.write();
            return prop;
        }
    );

    auto write_b = scheduler.make_functor(
        [=]
        {
            std::cout << "Write to B" << std::endl;
        },
        [=]
        {
            Scheduler::Properties prop;
            prop.policy<rmngr::ResourceUserPolicy>() += b.write();
            return prop;                      
        }
    );

    auto read_ab = scheduler.make_functor(
        [=]()
        {
            std::cout << "Read from A & B" << std::endl;
        },
        [=]
        {
            Scheduler::Properties prop;
            prop.policy<rmngr::ResourceUserPolicy>() += a.read();
            prop.policy<rmngr::ResourceUserPolicy>() += b.read();
            return prop;                      
        }
    );

    for(int i = 0; i < 1; ++i)
    {
        write_a();
        write_a();
        read_a();
        read_a();
        write_b();
        read_ab();
    }

    return 0;
}



#include <iostream>
#include <thread>
#include <chrono>

#include <rmngr/ioresource.hpp>
#include <rmngr/scheduling_context.hpp>
#include <rmngr/dependency_graph.hpp>

int main(void)
{
    rmngr::SchedulingContext<rmngr::DependencyGraph, 8, rmngr::ResourceUser::CheckDependency>context;
    rmngr::IOResource<> a;
    rmngr::IOResource<> b;

    auto read_a = context.make_functor([]()
    {
        std::cout << "Read from A" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }, {a.make_access({rmngr::IOAccess::read})}, "Ar");

    auto write_a = context.make_functor([]()
    {
        std::cout << "Write to A" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }, {a.make_access({rmngr::IOAccess::write})}, "Aw");

    auto write_b = context.make_functor([]()
    {
        std::cout << "Write to B" << std::endl;
    },
    {b.make_access({rmngr::IOAccess::write})}, "Bw");

    auto read_ab = context.make_functor([]()
    {
        std::cout << "Read from A & B" << std::endl;
    },
    {a.make_access({rmngr::IOAccess::read}), b.make_access({rmngr::IOAccess::read})}, "Ar, Br");

    for(int i = 0; i < 3; ++i)
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



#include <iostream>
#include <thread>
#include <chrono>

#include <rmngr/resource/ioresource.hpp>
#include <rmngr/property/resource.hpp>
#include <rmngr/property/inherit.hpp>
#include <rmngr/manager.hpp>

using Properties = rmngr::TaskProperties<
    rmngr::ResourceProperty
>;

int main(void)
{
    rmngr::Manager<
        Properties,
        rmngr::ResourceEnqueuePolicy
    > mgr( 4 );

    rmngr::IOResource a;
    rmngr::IOResource b;

    auto read_a = mgr.make_functor(
        [a]
        {
            std::cout << "Read from A" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        },
        [a]
        {
            return Properties::Builder().resources({ a.read() });
        }
    );

    auto write_a = mgr.make_functor(
        [a]
        {
            std::cout << "Write to A" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        },
        [a]
        {
            return Properties::Builder().resources({ a.write() });
        }
    );

    auto write_b = mgr.make_functor(
        [b]
        {
            std::cout << "Write to B" << std::endl;
        },
        [b]
        {
            return Properties::Builder().resources({ b.write() });
        }
    );

    auto read_ab = mgr.make_functor(
        [a,b]
        {
            std::cout << "Read from A & B" << std::endl;
        },
        [a,b]
        {
            return Properties::Builder().resources({ a.read(), b.read() });
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


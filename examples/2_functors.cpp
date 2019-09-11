
#include <iostream>

#include <rmngr/manager.hpp>

int fun1_impl (int x)
{
    return x*x;
}

int main()
{
    rmngr::Manager<> mgr( 1 /* number of threads */ );

    auto fun = mgr.make_functor(&fun1_impl);
    std::cout << "fun(2) = " << fun(2).get() << std::endl;

    return 0;
}


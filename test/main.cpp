#include <iostream>

#include <rmngr/queue.hpp>
#include <rmngr/resource.hpp>
#include <rmngr/functor.hpp>

int main(int argc, char* argv[])
{
    rmngr::Queue<rmmngr::Functor, rmmngr::Functor::CheckFunctor, rmmngr::Functor::Label> queue;

    rmngr::Resource a(1);
    rmngr::Resource b(2);

    FUNCTOR(functor1, a.read);
    FUNCTOR(functor2, a.write);
    FUNCTOR(functor3, a.read, a.write);
    FUNCTOR(functor4, b.read);
    FUNCTOR(functor5, b.write);
    FUNCTOR(functor6, b.read, b.write);
    FUNCTOR(functor7, a.write, b.write);

    functor1();
    functor4();
    functor6();
    functor7();
    functor3();
    functor1();
    functor2();
    functor5();
    functor7();

    queue.write_dependency_graph(std::cout);

    return 0;
}


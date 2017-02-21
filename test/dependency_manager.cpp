#include <boost/test/unit_test.hpp>

#include <iostream>
#include <rmngr/dependency_manager.hpp>

BOOST_AUTO_TEST_SUITE(resource);

using namespace rmngr;

BOOST_AUTO_TEST_CASE(dependency_manager)
{
    DependencyManager<int> m;

    DependencyManager<int>::VertexID root = m.add_vertex(1);
    DependencyManager<int>::VertexID read = m.add_vertex(2);
    DependencyManager<int>::VertexID write = m.add_vertex(3);
    DependencyManager<int>::VertexID aadd = m.add_vertex(4);
    DependencyManager<int>::VertexID amul = m.add_vertex(5);

    // state modifying operations
    m.add_dependency(root, read);
    m.add_dependency(root, write);
    m.add_dependency(root, aadd);
    m.add_dependency(root, amul);

    // state depending operations
    m.add_dependency(read, root);
    m.add_dependency(write, root);
    m.add_dependency(aadd, root);
    m.add_dependency(amul, root);

    // non atomic operations
    m.add_dependency(write, write);

    BOOST_CHECK( m.check_dependency(read, read) == false );
    BOOST_CHECK( m.check_dependency(read, write) == true );
    BOOST_CHECK( m.check_dependency(read, aadd) == true );
    BOOST_CHECK( m.check_dependency(read, amul) == true );

    BOOST_CHECK( m.check_dependency(write, read) == true );
    BOOST_CHECK( m.check_dependency(write, write) == true );
    BOOST_CHECK( m.check_dependency(write, aadd) == true );
    BOOST_CHECK( m.check_dependency(write, amul) == true );

    BOOST_CHECK( m.check_dependency(aadd, read) == true );
    BOOST_CHECK( m.check_dependency(aadd, write) == true );
    BOOST_CHECK( m.check_dependency(aadd, aadd) == false );
    BOOST_CHECK( m.check_dependency(aadd, amul) == true );

    BOOST_CHECK( m.check_dependency(amul, read) == true );
    BOOST_CHECK( m.check_dependency(amul, write) == true );
    BOOST_CHECK( m.check_dependency(amul, aadd) == true );
    BOOST_CHECK( m.check_dependency(amul, amul) == false );

}

BOOST_AUTO_TEST_SUITE_END();


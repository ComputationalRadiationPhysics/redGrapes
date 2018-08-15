
#include <catch/catch.hpp>

#include <boost/graph/adjacency_matrix.hpp>
#include <rmngr/access/dependency_manager.hpp>

TEST_CASE("dependency_manager")
{
    rmngr::DependencyManager< boost::adjacency_matrix<boost::undirectedS> > m(5);

    enum { root, read, write, aadd, amul };
    boost::add_edge(root, read, m.graph());
    boost::add_edge(root, write, m.graph());
    boost::add_edge(root, aadd, m.graph());
    boost::add_edge(root, amul, m.graph());
    boost::add_edge(write, write, m.graph());

    m.update();

    REQUIRE( m.is_serial(read, read) == false );
    REQUIRE( m.is_serial(read, write) == true );
    REQUIRE( m.is_serial(read, aadd) == true );
    REQUIRE( m.is_serial(read, amul) == true );

    REQUIRE( m.is_serial(write, read) == true );
    REQUIRE( m.is_serial(write, write) == true );
    REQUIRE( m.is_serial(write, aadd) == true );
    REQUIRE( m.is_serial(write, amul) == true );

    REQUIRE( m.is_serial(aadd, read) == true );
    REQUIRE( m.is_serial(aadd, write) == true );
    REQUIRE( m.is_serial(aadd, aadd) == false );
    REQUIRE( m.is_serial(aadd, amul) == true );

    REQUIRE( m.is_serial(amul, read) == true );
    REQUIRE( m.is_serial(amul, write) == true );
    REQUIRE( m.is_serial(amul, aadd) == true );
    REQUIRE( m.is_serial(amul, amul) == false );
}


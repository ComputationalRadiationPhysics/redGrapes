#include <boost/test/unit_test.hpp>

#include <rmngr/resource.hpp>
#include <rmngr/ioaccess.hpp>

BOOST_AUTO_TEST_SUITE(resource);

BOOST_AUTO_TEST_CASE(access)
{
    rmngr::StaticResource<rmngr::IOAccess> a;
    rmngr::StaticResource<rmngr::IOAccess> b;

    auto read_a = a.create_resource_access(rmngr::IOAccess::read);
    auto write_a = a.create_resource_access(rmngr::IOAccess::write);
    auto read_b = b.create_resource_access(rmngr::IOAccess::read);
    auto write_b = b.create_resource_access(rmngr::IOAccess::write);

    BOOST_CHECK( read_a.check_dependency(read_a) == false );
    BOOST_CHECK( read_a.check_dependency(write_a) == true );
    BOOST_CHECK( write_a.check_dependency(read_a) == true );

    BOOST_CHECK( read_b.check_dependency(read_b) == false );
    BOOST_CHECK( read_b.check_dependency(write_b) == true );
    BOOST_CHECK( write_b.check_dependency(read_b) == true );

    BOOST_CHECK( read_a.check_dependency(read_b) == false );
    BOOST_CHECK( read_a.check_dependency(write_b) == false );
    BOOST_CHECK( write_a.check_dependency(read_b) == false );

    BOOST_CHECK( read_b.check_dependency(read_a) == false );
    BOOST_CHECK( read_b.check_dependency(write_a) == false );
    BOOST_CHECK( write_b.check_dependency(read_a) == false );
}

BOOST_AUTO_TEST_SUITE_END();


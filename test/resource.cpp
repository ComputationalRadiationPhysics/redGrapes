
#include <catch/catch.hpp>

#include <rmngr/resource/resource.hpp>
#include <rmngr/resource/ioresource.hpp>

struct Access
{
    static bool is_serial(Access a, Access b)
    { return true; }

    bool is_superset_of(Access a) const
    { return true; }

    bool operator==(Access const & other) const
    { return false; }

    friend std::ostream& operator<< (std::ostream & out, Access const &)
    { return out; }
};

TEST_CASE("Resource ID")
{
    rmngr::Resource< Access > a, b;

    // same resource
    REQUIRE( rmngr::ResourceAccess::is_serial( a.make_access(Access{}), a.make_access(Access{}) ) == true );
    REQUIRE( rmngr::ResourceAccess::is_serial( b.make_access(Access{}), b.make_access(Access{}) ) == true );

    // same resource, but copied
    rmngr::Resource< Access > a2(a);
    REQUIRE( rmngr::ResourceAccess::is_serial( a.make_access(Access{}), a2.make_access(Access{}) ) == true );

    // different resource
    REQUIRE( rmngr::ResourceAccess::is_serial( a.make_access(Access{}), b.make_access(Access{}) ) == false );
    REQUIRE( rmngr::ResourceAccess::is_serial( b.make_access(Access{}), a.make_access(Access{}) ) == false );
}

TEST_CASE("IOResource")
{
    rmngr::IOResource a,b;

    REQUIRE( rmngr::ResourceAccess::is_serial(a.read(), a.read()) == false );
    REQUIRE( rmngr::ResourceAccess::is_serial(a.read(), a.write()) == true );
    REQUIRE( rmngr::ResourceAccess::is_serial(a.write(), a.read()) == true );
    REQUIRE( rmngr::ResourceAccess::is_serial(a.write(), a.write()) == true );

    REQUIRE( rmngr::ResourceAccess::is_serial(b.read(), b.read()) == false );
    REQUIRE( rmngr::ResourceAccess::is_serial(b.read(), b.write()) == true );
    REQUIRE( rmngr::ResourceAccess::is_serial(b.write(), b.read()) == true );
    REQUIRE( rmngr::ResourceAccess::is_serial(b.write(), b.write()) == true );

    REQUIRE( rmngr::ResourceAccess::is_serial(a.read(), b.read()) == false );
    REQUIRE( rmngr::ResourceAccess::is_serial(a.read(), b.write()) == false );
    REQUIRE( rmngr::ResourceAccess::is_serial(a.write(), b.read()) == false );
    REQUIRE( rmngr::ResourceAccess::is_serial(a.write(), b.write()) == false );

    REQUIRE( rmngr::ResourceAccess::is_serial(b.read(), a.read()) == false );
    REQUIRE( rmngr::ResourceAccess::is_serial(b.read(), a.write()) == false );
    REQUIRE( rmngr::ResourceAccess::is_serial(b.write(), a.read()) == false );
    REQUIRE( rmngr::ResourceAccess::is_serial(b.write(), a.write()) == false );
}


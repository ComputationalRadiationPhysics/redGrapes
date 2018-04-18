
#include <catch/catch.hpp>

#include <rmngr/ioresource.hpp>
#include <rmngr/resource_user.hpp>

TEST_CASE("Resource User")
{
    rmngr::IOResource a, b;

    rmngr::ResourceUser f1({a.read()});
    rmngr::ResourceUser f2({a.read(), a.write()});
    rmngr::ResourceUser f3({b.read()});
    rmngr::ResourceUser f4({b.read(), b.write()});
    rmngr::ResourceUser f5({a.read(), a.write(), b.read(), b.write()});

    REQUIRE( rmngr::ResourceUser::is_serial(f1, f1) == false );
    REQUIRE( rmngr::ResourceUser::is_serial(f1, f2) == true );
    REQUIRE( rmngr::ResourceUser::is_serial(f1, f3) == false );
    REQUIRE( rmngr::ResourceUser::is_serial(f1, f4) == false );
    REQUIRE( rmngr::ResourceUser::is_serial(f1, f5) == true );

    REQUIRE( rmngr::ResourceUser::is_serial(f2, f1) == true );
    REQUIRE( rmngr::ResourceUser::is_serial(f2, f2) == true );
    REQUIRE( rmngr::ResourceUser::is_serial(f2, f3) == false );
    REQUIRE( rmngr::ResourceUser::is_serial(f2, f4) == false );
    REQUIRE( rmngr::ResourceUser::is_serial(f2, f5) == true );

    REQUIRE( rmngr::ResourceUser::is_serial(f3, f1) == false );
    REQUIRE( rmngr::ResourceUser::is_serial(f3, f2) == false );
    REQUIRE( rmngr::ResourceUser::is_serial(f3, f3) == false );
    REQUIRE( rmngr::ResourceUser::is_serial(f3, f4) == true );
    REQUIRE( rmngr::ResourceUser::is_serial(f3, f5) == true );

    REQUIRE( rmngr::ResourceUser::is_serial(f4, f1) == false );
    REQUIRE( rmngr::ResourceUser::is_serial(f4, f2) == false );
    REQUIRE( rmngr::ResourceUser::is_serial(f4, f3) == true );
    REQUIRE( rmngr::ResourceUser::is_serial(f4, f4) == true );
    REQUIRE( rmngr::ResourceUser::is_serial(f4, f5) == true );

    REQUIRE( rmngr::ResourceUser::is_serial(f5, f1) == true );
    REQUIRE( rmngr::ResourceUser::is_serial(f5, f2) == true );
    REQUIRE( rmngr::ResourceUser::is_serial(f5, f3) == true );
    REQUIRE( rmngr::ResourceUser::is_serial(f5, f4) == true );
    REQUIRE( rmngr::ResourceUser::is_serial(f5, f5) == true );
}


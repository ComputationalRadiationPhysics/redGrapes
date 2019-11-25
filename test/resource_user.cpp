
#include <catch/catch.hpp>

#include <redGrapes/resource/ioresource.hpp>
#include <redGrapes/resource/resource_user.hpp>

TEST_CASE("Resource User")
{
    redGrapes::IOResource a, b;

    redGrapes::ResourceUser f1({a.read()});
    redGrapes::ResourceUser f2({a.read(), a.write()});
    redGrapes::ResourceUser f3({b.read()});
    redGrapes::ResourceUser f4({b.read(), b.write()});
    redGrapes::ResourceUser f5({a.read(), a.write(), b.read(), b.write()});

    REQUIRE( redGrapes::ResourceUser::is_serial(f1, f1) == false );
    REQUIRE( redGrapes::ResourceUser::is_serial(f1, f2) == true );
    REQUIRE( redGrapes::ResourceUser::is_serial(f1, f3) == false );
    REQUIRE( redGrapes::ResourceUser::is_serial(f1, f4) == false );
    REQUIRE( redGrapes::ResourceUser::is_serial(f1, f5) == true );

    REQUIRE( redGrapes::ResourceUser::is_serial(f2, f1) == true );
    REQUIRE( redGrapes::ResourceUser::is_serial(f2, f2) == true );
    REQUIRE( redGrapes::ResourceUser::is_serial(f2, f3) == false );
    REQUIRE( redGrapes::ResourceUser::is_serial(f2, f4) == false );
    REQUIRE( redGrapes::ResourceUser::is_serial(f2, f5) == true );

    REQUIRE( redGrapes::ResourceUser::is_serial(f3, f1) == false );
    REQUIRE( redGrapes::ResourceUser::is_serial(f3, f2) == false );
    REQUIRE( redGrapes::ResourceUser::is_serial(f3, f3) == false );
    REQUIRE( redGrapes::ResourceUser::is_serial(f3, f4) == true );
    REQUIRE( redGrapes::ResourceUser::is_serial(f3, f5) == true );

    REQUIRE( redGrapes::ResourceUser::is_serial(f4, f1) == false );
    REQUIRE( redGrapes::ResourceUser::is_serial(f4, f2) == false );
    REQUIRE( redGrapes::ResourceUser::is_serial(f4, f3) == true );
    REQUIRE( redGrapes::ResourceUser::is_serial(f4, f4) == true );
    REQUIRE( redGrapes::ResourceUser::is_serial(f4, f5) == true );

    REQUIRE( redGrapes::ResourceUser::is_serial(f5, f1) == true );
    REQUIRE( redGrapes::ResourceUser::is_serial(f5, f2) == true );
    REQUIRE( redGrapes::ResourceUser::is_serial(f5, f3) == true );
    REQUIRE( redGrapes::ResourceUser::is_serial(f5, f4) == true );
    REQUIRE( redGrapes::ResourceUser::is_serial(f5, f5) == true );


    REQUIRE( f1.is_superset_of(f1) == true );
    REQUIRE( f1.is_superset_of(f2) == false );
    REQUIRE( f1.is_superset_of(f3) == false );
    REQUIRE( f1.is_superset_of(f4) == false );
    REQUIRE( f1.is_superset_of(f5) == false );

    REQUIRE( f2.is_superset_of(f1) == true );
    REQUIRE( f2.is_superset_of(f2) == true );
    REQUIRE( f2.is_superset_of(f3) == false );
    REQUIRE( f2.is_superset_of(f4) == false );
    REQUIRE( f2.is_superset_of(f5) == false );
}


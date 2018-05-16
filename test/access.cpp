
#include <catch/catch.hpp>

#include <rmngr/access/io.hpp>
#include <rmngr/access/area.hpp>
#include <rmngr/access/combine.hpp>
#include <rmngr/access/field.hpp>

using namespace rmngr::access;

TEST_CASE("IOAccess")
{
    REQUIRE( IOAccess::is_serial( IOAccess{IOAccess::read}, IOAccess{IOAccess::read} ) == false );
    REQUIRE( IOAccess::is_serial( IOAccess{IOAccess::read}, IOAccess{IOAccess::write} ) == true );
    REQUIRE( IOAccess::is_serial( IOAccess{IOAccess::read}, IOAccess{IOAccess::aadd} ) == true );
    REQUIRE( IOAccess::is_serial( IOAccess{IOAccess::read}, IOAccess{IOAccess::amul} ) == true );

    REQUIRE( IOAccess::is_serial( IOAccess{IOAccess::write}, IOAccess{IOAccess::read} ) == true );
    REQUIRE( IOAccess::is_serial( IOAccess{IOAccess::write}, IOAccess{IOAccess::write} ) == true );
    REQUIRE( IOAccess::is_serial( IOAccess{IOAccess::write}, IOAccess{IOAccess::aadd} ) == true );
    REQUIRE( IOAccess::is_serial( IOAccess{IOAccess::write}, IOAccess{IOAccess::amul} ) == true );

    REQUIRE( IOAccess::is_serial( IOAccess{IOAccess::aadd}, IOAccess{IOAccess::read} ) == true );
    REQUIRE( IOAccess::is_serial( IOAccess{IOAccess::aadd}, IOAccess{IOAccess::write} ) == true );
    REQUIRE( IOAccess::is_serial( IOAccess{IOAccess::aadd}, IOAccess{IOAccess::aadd} ) == false );
    REQUIRE( IOAccess::is_serial( IOAccess{IOAccess::aadd}, IOAccess{IOAccess::amul} ) == true );

    REQUIRE( IOAccess::is_serial( IOAccess{IOAccess::amul}, IOAccess{IOAccess::read} ) == true );
    REQUIRE( IOAccess::is_serial( IOAccess{IOAccess::amul}, IOAccess{IOAccess::write} ) == true );
    REQUIRE( IOAccess::is_serial( IOAccess{IOAccess::amul}, IOAccess{IOAccess::aadd} ) == true );
    REQUIRE( IOAccess::is_serial( IOAccess{IOAccess::amul}, IOAccess{IOAccess::amul} ) == false );
}

TEST_CASE("AreaAccess")
{
    // --[-----]--(-----)--
    REQUIRE( AreaAccess::is_serial( AreaAccess({10, 20}), AreaAccess({30, 40}) ) == false );
    // --(-----)--[-----]--
    REQUIRE( AreaAccess::is_serial( AreaAccess({30, 40}), AreaAccess({10, 20}) ) == false );

    // --[--(--]--)--
    REQUIRE( AreaAccess::is_serial( AreaAccess({10, 20}), AreaAccess({15, 25}) ) == true );
    // --(--[--)--]--
    REQUIRE( AreaAccess::is_serial( AreaAccess({15, 25}), AreaAccess({10, 20}) ) == true );

    // --[--(--)--]--
    REQUIRE( AreaAccess::is_serial( AreaAccess({10, 30}), AreaAccess({15, 25}) ) == true );
    // --(--[--]--)--
    REQUIRE( AreaAccess::is_serial( AreaAccess({15, 25}), AreaAccess({10, 30}) ) == true );
}

TEST_CASE("CombineAccess")
{
    using A = CombineAccess<
        IOAccess,
        AreaAccess,
        And_t
      >;

    REQUIRE(A::is_serial(
                A(IOAccess{IOAccess::read}, AreaAccess({10, 20})),
                A(IOAccess{IOAccess::read}, AreaAccess({15, 25})))
            == false);

    REQUIRE(A::is_serial(
                A(IOAccess{IOAccess::read}, AreaAccess({10, 20})),
                A(IOAccess{IOAccess::write}, AreaAccess({15, 25})))
            == true);

    REQUIRE(A::is_serial(
                A(IOAccess{IOAccess::read}, AreaAccess({10, 20})),
                A(IOAccess{IOAccess::write}, AreaAccess({30, 40})))
            == false);

    using B = CombineAccess<
        IOAccess,
        AreaAccess,
        Or_t
    >;

    REQUIRE(B::is_serial(
                B(IOAccess{IOAccess::read}, AreaAccess({10, 20})),
                B(IOAccess{IOAccess::read}, AreaAccess({30, 40})))
            == false);

    REQUIRE(B::is_serial(
                B(IOAccess{IOAccess::read}, AreaAccess({10, 20})),
                B(IOAccess{IOAccess::read}, AreaAccess({15, 25})))
            == true);

    REQUIRE(B::is_serial(
                B(IOAccess{IOAccess::read}, AreaAccess({10, 20})),
                B(IOAccess{IOAccess::write}, AreaAccess({15, 25})))
            == true);

    REQUIRE(B::is_serial(
                B(IOAccess{IOAccess::read}, AreaAccess({10, 20})),
                B(IOAccess{IOAccess::write}, AreaAccess({30, 40})))
            == true);

}

TEST_CASE("ArrayAccess")
{
    using A = ArrayAccess<
        IOAccess,
        2,
        And_t
    >;

    REQUIRE(A::is_serial(
                A({ IOAccess{IOAccess::read}, IOAccess{IOAccess::read} }),
                A({ IOAccess{IOAccess::read}, IOAccess{IOAccess::read} }))
            == false);

    REQUIRE(A::is_serial(
                A({ IOAccess{IOAccess::read}, IOAccess{IOAccess::write} }),
                A({ IOAccess{IOAccess::read}, IOAccess{IOAccess::read} }))
            == false);

    REQUIRE(A::is_serial(
                A({ IOAccess{IOAccess::write}, IOAccess{IOAccess::write} }),
                A({ IOAccess{IOAccess::read}, IOAccess{IOAccess::read} }))
            == true);


    using B = ArrayAccess<
        IOAccess,
        2,
        Or_t
    >;


    REQUIRE(B::is_serial(
                B({ IOAccess{IOAccess::read}, IOAccess{IOAccess::read} }),
                B({ IOAccess{IOAccess::read}, IOAccess{IOAccess::read} }))
            == false);

    REQUIRE(B::is_serial(
                B({ IOAccess{IOAccess::read}, IOAccess{IOAccess::write} }),
                B({ IOAccess{IOAccess::read}, IOAccess{IOAccess::read} }))
            == true);

    REQUIRE(B::is_serial(
                B({ IOAccess{IOAccess::write}, IOAccess{IOAccess::write} }),
                B({ IOAccess{IOAccess::read}, IOAccess{IOAccess::read} }))
            == true);
}

TEST_CASE("FieldAccess")
{
    using Arr = ArrayAccess<AreaAccess, 3, And_t>;
    REQUIRE(FieldAccess<3>::is_serial(
                FieldAccess<3>(
                    IOAccess{IOAccess::read},
                    Arr({
                        AreaAccess({0,10}),
                        AreaAccess({0,10}),
                        AreaAccess({0,10})})),
                FieldAccess<3>(
                    IOAccess{IOAccess::read},
                    Arr({
                        AreaAccess({0,10}),
                        AreaAccess({0,10}),
                        AreaAccess({0,10})})))
            == false);

    REQUIRE(FieldAccess<3>::is_serial(
                FieldAccess<3>(
                    IOAccess{IOAccess::write},
                    Arr({
                        AreaAccess({0,10}),
                        AreaAccess({0,10}),
                        AreaAccess({0,10})})),
                FieldAccess<3>(
                    IOAccess{IOAccess::read},
                    Arr({
                        AreaAccess({0,10}),
                        AreaAccess({0,10}),
                        AreaAccess({0,10})})))
            == true);
}


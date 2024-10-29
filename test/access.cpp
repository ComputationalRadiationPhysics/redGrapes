
#include <redGrapes/resource/access/combine.hpp>
#include <redGrapes/resource/access/field.hpp>
#include <redGrapes/resource/access/io.hpp>
#include <redGrapes/resource/access/range.hpp>

#include <catch2/catch_test_macros.hpp>

using namespace redGrapes::access;

TEST_CASE("IOAccess")
{
    REQUIRE(IOAccess::is_serial(IOAccess{IOAccess::read}, IOAccess{IOAccess::read}) == false);
    REQUIRE(IOAccess::is_serial(IOAccess{IOAccess::read}, IOAccess{IOAccess::write}) == true);
    REQUIRE(IOAccess::is_serial(IOAccess{IOAccess::read}, IOAccess{IOAccess::aadd}) == true);
    REQUIRE(IOAccess::is_serial(IOAccess{IOAccess::read}, IOAccess{IOAccess::amul}) == true);

    REQUIRE(IOAccess::is_serial(IOAccess{IOAccess::write}, IOAccess{IOAccess::read}) == true);
    REQUIRE(IOAccess::is_serial(IOAccess{IOAccess::write}, IOAccess{IOAccess::write}) == true);
    REQUIRE(IOAccess::is_serial(IOAccess{IOAccess::write}, IOAccess{IOAccess::aadd}) == true);
    REQUIRE(IOAccess::is_serial(IOAccess{IOAccess::write}, IOAccess{IOAccess::amul}) == true);

    REQUIRE(IOAccess::is_serial(IOAccess{IOAccess::aadd}, IOAccess{IOAccess::read}) == true);
    REQUIRE(IOAccess::is_serial(IOAccess{IOAccess::aadd}, IOAccess{IOAccess::write}) == true);
    REQUIRE(IOAccess::is_serial(IOAccess{IOAccess::aadd}, IOAccess{IOAccess::aadd}) == false);
    REQUIRE(IOAccess::is_serial(IOAccess{IOAccess::aadd}, IOAccess{IOAccess::amul}) == true);

    REQUIRE(IOAccess::is_serial(IOAccess{IOAccess::amul}, IOAccess{IOAccess::read}) == true);
    REQUIRE(IOAccess::is_serial(IOAccess{IOAccess::amul}, IOAccess{IOAccess::write}) == true);
    REQUIRE(IOAccess::is_serial(IOAccess{IOAccess::amul}, IOAccess{IOAccess::aadd}) == true);
    REQUIRE(IOAccess::is_serial(IOAccess{IOAccess::amul}, IOAccess{IOAccess::amul}) == false);

    // subsets
    REQUIRE(IOAccess{IOAccess::read}.is_superset_of(IOAccess{IOAccess::read}) == true);
    REQUIRE(IOAccess{IOAccess::read}.is_superset_of(IOAccess{IOAccess::write}) == false);
    REQUIRE(IOAccess{IOAccess::read}.is_superset_of(IOAccess{IOAccess::aadd}) == false);
    REQUIRE(IOAccess{IOAccess::read}.is_superset_of(IOAccess{IOAccess::amul}) == false);

    REQUIRE(IOAccess{IOAccess::write}.is_superset_of(IOAccess{IOAccess::read}) == true);
    REQUIRE(IOAccess{IOAccess::write}.is_superset_of(IOAccess{IOAccess::write}) == true);
    REQUIRE(IOAccess{IOAccess::write}.is_superset_of(IOAccess{IOAccess::aadd}) == true);
    REQUIRE(IOAccess{IOAccess::write}.is_superset_of(IOAccess{IOAccess::amul}) == true);

    REQUIRE(IOAccess{IOAccess::aadd}.is_superset_of(IOAccess{IOAccess::read}) == false);
    REQUIRE(IOAccess{IOAccess::aadd}.is_superset_of(IOAccess{IOAccess::write}) == false);
    REQUIRE(IOAccess{IOAccess::aadd}.is_superset_of(IOAccess{IOAccess::aadd}) == true);
    REQUIRE(IOAccess{IOAccess::aadd}.is_superset_of(IOAccess{IOAccess::amul}) == false);

    REQUIRE(IOAccess{IOAccess::amul}.is_superset_of(IOAccess{IOAccess::read}) == false);
    REQUIRE(IOAccess{IOAccess::amul}.is_superset_of(IOAccess{IOAccess::write}) == false);
    REQUIRE(IOAccess{IOAccess::amul}.is_superset_of(IOAccess{IOAccess::aadd}) == false);
    REQUIRE(IOAccess{IOAccess::amul}.is_superset_of(IOAccess{IOAccess::amul}) == true);
}

TEST_CASE("RangeAccess")
{
    // --[-----]--(-----)--
    REQUIRE(RangeAccess::is_serial(RangeAccess({10, 20}), RangeAccess({30, 40})) == false);
    REQUIRE(RangeAccess({10, 20}).is_superset_of(RangeAccess({30, 40})) == false);
    // --(-----)--[-----]--
    REQUIRE(RangeAccess::is_serial(RangeAccess({30, 40}), RangeAccess({10, 20})) == false);
    REQUIRE(RangeAccess({30, 40}).is_superset_of(RangeAccess({10, 20})) == false);

    // --[--(--]--)--
    REQUIRE(RangeAccess::is_serial(RangeAccess({10, 20}), RangeAccess({15, 25})) == true);
    REQUIRE(RangeAccess({10, 20}).is_superset_of(RangeAccess({15, 25})) == false);
    // --(--[--)--]--
    REQUIRE(RangeAccess::is_serial(RangeAccess({15, 25}), RangeAccess({10, 20})) == true);
    REQUIRE(RangeAccess({15, 15}).is_superset_of(RangeAccess({10, 20})) == false);

    // --[--(--)--]--
    REQUIRE(RangeAccess::is_serial(RangeAccess({10, 30}), RangeAccess({15, 25})) == true);
    REQUIRE(RangeAccess({10, 30}).is_superset_of(RangeAccess({15, 25})) == true);
    // --(--[--]--)--
    REQUIRE(RangeAccess::is_serial(RangeAccess({15, 25}), RangeAccess({10, 30})) == true);
    REQUIRE(RangeAccess({15, 25}).is_superset_of(RangeAccess({10, 30})) == false);
}

TEST_CASE("CombineAccess")
{
    using A = CombineAccess<IOAccess, RangeAccess, And_t>;

    REQUIRE(
        A::is_serial(
            A(IOAccess{IOAccess::read}, RangeAccess({10, 20})),
            A(IOAccess{IOAccess::read}, RangeAccess({15, 25})))
        == false);

    REQUIRE(
        A::is_serial(
            A(IOAccess{IOAccess::read}, RangeAccess({10, 20})),
            A(IOAccess{IOAccess::write}, RangeAccess({15, 25})))
        == true);

    REQUIRE(
        A::is_serial(
            A(IOAccess{IOAccess::read}, RangeAccess({10, 20})),
            A(IOAccess{IOAccess::write}, RangeAccess({30, 40})))
        == false);

    REQUIRE(
        A(IOAccess{IOAccess::read}, RangeAccess({10, 20}))
            .is_superset_of(A(IOAccess{IOAccess::read}, RangeAccess({15, 25})))
        == false);

    REQUIRE(
        A(IOAccess{IOAccess::write}, RangeAccess({10, 30}))
            .is_superset_of(A(IOAccess{IOAccess::read}, RangeAccess({15, 25})))
        == true);

    using B = CombineAccess<IOAccess, RangeAccess, Or_t>;

    REQUIRE(
        B::is_serial(
            B(IOAccess{IOAccess::read}, RangeAccess({10, 20})),
            B(IOAccess{IOAccess::read}, RangeAccess({30, 40})))
        == false);

    REQUIRE(
        B::is_serial(
            B(IOAccess{IOAccess::read}, RangeAccess({10, 20})),
            B(IOAccess{IOAccess::read}, RangeAccess({15, 25})))
        == true);

    REQUIRE(
        B::is_serial(
            B(IOAccess{IOAccess::read}, RangeAccess({10, 20})),
            B(IOAccess{IOAccess::write}, RangeAccess({15, 25})))
        == true);

    REQUIRE(
        B::is_serial(
            B(IOAccess{IOAccess::read}, RangeAccess({10, 20})),
            B(IOAccess{IOAccess::write}, RangeAccess({30, 40})))
        == true);
}

TEST_CASE("ArrayAccess")
{
    using A = ArrayAccess<IOAccess, 2, And_t>;

    REQUIRE(
        A::is_serial(
            A({IOAccess{IOAccess::read}, IOAccess{IOAccess::read}}),
            A({IOAccess{IOAccess::read}, IOAccess{IOAccess::read}}))
        == false);

    REQUIRE(
        A::is_serial(
            A({IOAccess{IOAccess::read}, IOAccess{IOAccess::write}}),
            A({IOAccess{IOAccess::read}, IOAccess{IOAccess::read}}))
        == false);

    REQUIRE(
        A::is_serial(
            A({IOAccess{IOAccess::write}, IOAccess{IOAccess::write}}),
            A({IOAccess{IOAccess::read}, IOAccess{IOAccess::read}}))
        == true);

    REQUIRE(
        A({IOAccess{IOAccess::read}, IOAccess{IOAccess::write}})
            .is_superset_of(A({IOAccess{IOAccess::read}, IOAccess{IOAccess::read}}))
        == true);

    REQUIRE(
        A({IOAccess{IOAccess::read}, IOAccess{IOAccess::write}})
            .is_superset_of(A({IOAccess{IOAccess::write}, IOAccess{IOAccess::read}}))
        == false);

    using B = ArrayAccess<IOAccess, 2, Or_t>;


    REQUIRE(
        B::is_serial(
            B({IOAccess{IOAccess::read}, IOAccess{IOAccess::read}}),
            B({IOAccess{IOAccess::read}, IOAccess{IOAccess::read}}))
        == false);

    REQUIRE(
        B::is_serial(
            B({IOAccess{IOAccess::read}, IOAccess{IOAccess::write}}),
            B({IOAccess{IOAccess::read}, IOAccess{IOAccess::read}}))
        == true);

    REQUIRE(
        B::is_serial(
            B({IOAccess{IOAccess::write}, IOAccess{IOAccess::write}}),
            B({IOAccess{IOAccess::read}, IOAccess{IOAccess::read}}))
        == true);
}

TEST_CASE("FieldAccess")
{
    using Arr = ArrayAccess<RangeAccess, 3, And_t>;
    REQUIRE(
        FieldAccess<3>::is_serial(
            FieldAccess<3>(
                IOAccess{IOAccess::read},
                Arr({RangeAccess({0, 10}), RangeAccess({0, 10}), RangeAccess({0, 10})})),
            FieldAccess<3>(
                IOAccess{IOAccess::read},
                Arr({RangeAccess({0, 10}), RangeAccess({0, 10}), RangeAccess({0, 10})})))
        == false);

    REQUIRE(
        FieldAccess<3>::is_serial(
            FieldAccess<3>(
                IOAccess{IOAccess::write},
                Arr({RangeAccess({0, 10}), RangeAccess({0, 10}), RangeAccess({0, 10})})),
            FieldAccess<3>(
                IOAccess{IOAccess::read},
                Arr({RangeAccess({0, 10}), RangeAccess({0, 10}), RangeAccess({0, 10})})))
        == true);
}

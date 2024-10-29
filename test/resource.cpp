#include <redGrapes/redGrapes.hpp>
#include <redGrapes/resource/ioresource.hpp>
#include <redGrapes/resource/resource.hpp>

#include <catch2/catch_test_macros.hpp>

struct Access
{
    static bool is_serial([[maybe_unused]] Access a, [[maybe_unused]] Access b)
    {
        return true;
    }

    bool is_synchronizing() const
    {
        return true;
    }

    bool is_superset_of([[maybe_unused]] Access a) const
    {
        return true;
    }

    bool operator==([[maybe_unused]] Access const& other) const
    {
        return false;
    }
};

template<>
struct fmt::formatter<Access>
{
    constexpr auto parse(format_parse_context& ctx)
    {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format([[maybe_unused]] Access const& acc, FormatContext& ctx) const
    {
        return fmt::format_to(ctx.out(), "Access");
    }
};

TEST_CASE("Resource ID")
{
    auto rg = redGrapes::init(1);
    auto a = rg.createResource<Access>();
    auto b = rg.createResource<Access>();

    // same resource
    REQUIRE(redGrapes::is_serial(a.make_access(Access{}), a.make_access(Access{})) == true);
    REQUIRE(redGrapes::is_serial(b.make_access(Access{}), b.make_access(Access{})) == true);

    // same resource, but copied
    redGrapes::Resource<Access> a2(a);
    REQUIRE(redGrapes::is_serial(a.make_access(Access{}), a2.make_access(Access{})) == true);

    // different resource
    REQUIRE(redGrapes::is_serial(a.make_access(Access{}), b.make_access(Access{})) == false);
    REQUIRE(redGrapes::is_serial(b.make_access(Access{}), a.make_access(Access{})) == false);
}

TEST_CASE("IOResource")
{
    auto rg = redGrapes::init(1);

    redGrapes::IOResource<int> a, b;

    REQUIRE(redGrapes::is_serial(a.read(), a.read()) == false);
    REQUIRE(redGrapes::is_serial(a.read(), a.write()) == true);
    REQUIRE(redGrapes::is_serial(a.write(), a.read()) == true);
    REQUIRE(redGrapes::is_serial(a.write(), a.write()) == true);

    REQUIRE(redGrapes::is_serial(b.read(), b.read()) == false);
    REQUIRE(redGrapes::is_serial(b.read(), b.write()) == true);
    REQUIRE(redGrapes::is_serial(b.write(), b.read()) == true);
    REQUIRE(redGrapes::is_serial(b.write(), b.write()) == true);

    REQUIRE(redGrapes::is_serial(a.read(), b.read()) == false);
    REQUIRE(redGrapes::is_serial(a.read(), b.write()) == false);
    REQUIRE(redGrapes::is_serial(a.write(), b.read()) == false);
    REQUIRE(redGrapes::is_serial(a.write(), b.write()) == false);

    REQUIRE(redGrapes::is_serial(b.read(), a.read()) == false);
    REQUIRE(redGrapes::is_serial(b.read(), a.write()) == false);
    REQUIRE(redGrapes::is_serial(b.write(), a.read()) == false);
    REQUIRE(redGrapes::is_serial(b.write(), a.write()) == false);
}

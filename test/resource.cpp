
#include <catch/catch.hpp>

#include <redGrapes/resource/resource.hpp>
#include <redGrapes/resource/ioresource.hpp>

struct Access
{
    static bool is_serial(Access a, Access b)
    { return true; }

    bool is_superset_of(Access a) const
    { return true; }

    bool operator==(Access const & other) const
    { return false; }
};

template<>
struct fmt::formatter<Access>
{
    constexpr auto parse(format_parse_context& ctx)
    {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(Access const& acc, FormatContext& ctx)
    {
        return fmt::format_to(ctx.out(), "Access");
    }
};

TEST_CASE("Resource ID")
{
    redGrapes::Resource< Access > a, b;

    // same resource
    REQUIRE( redGrapes::ResourceAccess::is_serial( a.make_access(Access{}), a.make_access(Access{}) ) == true );
    REQUIRE( redGrapes::ResourceAccess::is_serial( b.make_access(Access{}), b.make_access(Access{}) ) == true );

    // same resource, but copied
    redGrapes::Resource< Access > a2(a);
    REQUIRE( redGrapes::ResourceAccess::is_serial( a.make_access(Access{}), a2.make_access(Access{}) ) == true );

    // different resource
    REQUIRE( redGrapes::ResourceAccess::is_serial( a.make_access(Access{}), b.make_access(Access{}) ) == false );
    REQUIRE( redGrapes::ResourceAccess::is_serial( b.make_access(Access{}), a.make_access(Access{}) ) == false );
}

TEST_CASE("IOResource")
{
    redGrapes::IOResource<int> a, b;

    REQUIRE( redGrapes::ResourceAccess::is_serial(a.read(), a.read()) == false );
    REQUIRE( redGrapes::ResourceAccess::is_serial(a.read(), a.write()) == true );
    REQUIRE( redGrapes::ResourceAccess::is_serial(a.write(), a.read()) == true );
    REQUIRE( redGrapes::ResourceAccess::is_serial(a.write(), a.write()) == true );

    REQUIRE( redGrapes::ResourceAccess::is_serial(b.read(), b.read()) == false );
    REQUIRE( redGrapes::ResourceAccess::is_serial(b.read(), b.write()) == true );
    REQUIRE( redGrapes::ResourceAccess::is_serial(b.write(), b.read()) == true );
    REQUIRE( redGrapes::ResourceAccess::is_serial(b.write(), b.write()) == true );

    REQUIRE( redGrapes::ResourceAccess::is_serial(a.read(), b.read()) == false );
    REQUIRE( redGrapes::ResourceAccess::is_serial(a.read(), b.write()) == false );
    REQUIRE( redGrapes::ResourceAccess::is_serial(a.write(), b.read()) == false );
    REQUIRE( redGrapes::ResourceAccess::is_serial(a.write(), b.write()) == false );

    REQUIRE( redGrapes::ResourceAccess::is_serial(b.read(), a.read()) == false );
    REQUIRE( redGrapes::ResourceAccess::is_serial(b.read(), a.write()) == false );
    REQUIRE( redGrapes::ResourceAccess::is_serial(b.write(), a.read()) == false );
    REQUIRE( redGrapes::ResourceAccess::is_serial(b.write(), a.write()) == false );
}


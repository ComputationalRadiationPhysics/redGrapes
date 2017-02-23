#include <boost/test/unit_test.hpp>

#include <rmngr/resource.hpp>
#include <rmngr/ioaccess.hpp>
#include <rmngr/resource_user.hpp>

BOOST_AUTO_TEST_SUITE(resource_user);

BOOST_AUTO_TEST_CASE(resource_users)
{
    typedef rmngr::StaticResource<rmngr::IOAccess> IOResource;
    typedef typename rmngr::ResourceUser::CheckDependency DCheck;

    IOResource a;
    IOResource b;

    rmngr::ResourceUser f1({a.make_access({rmngr::IOAccess::read})});
    rmngr::ResourceUser f2({a.make_access({rmngr::IOAccess::read, rmngr::IOAccess::write})});
    rmngr::ResourceUser f3({b.make_access({rmngr::IOAccess::read})});
    rmngr::ResourceUser f4({b.make_access({rmngr::IOAccess::read, rmngr::IOAccess::write})});
    rmngr::ResourceUser f5({a.make_access({rmngr::IOAccess::read, rmngr::IOAccess::write}),
                            b.make_access({rmngr::IOAccess::read, rmngr::IOAccess::write})
                           });

    BOOST_CHECK( DCheck::check(f1, f1) == false );
    BOOST_CHECK( DCheck::check(f1, f2) == true );
    BOOST_CHECK( DCheck::check(f1, f3) == false );
    BOOST_CHECK( DCheck::check(f1, f4) == false );
    BOOST_CHECK( DCheck::check(f1, f5) == true );

    BOOST_CHECK( DCheck::check(f2, f1) == true );
    BOOST_CHECK( DCheck::check(f2, f2) == true );
    BOOST_CHECK( DCheck::check(f2, f3) == false );
    BOOST_CHECK( DCheck::check(f2, f4) == false );
    BOOST_CHECK( DCheck::check(f2, f5) == true );

    BOOST_CHECK( DCheck::check(f3, f1) == false );
    BOOST_CHECK( DCheck::check(f3, f2) == false );
    BOOST_CHECK( DCheck::check(f3, f3) == false );
    BOOST_CHECK( DCheck::check(f3, f4) == true );
    BOOST_CHECK( DCheck::check(f3, f5) == true );

    BOOST_CHECK( DCheck::check(f4, f1) == false );
    BOOST_CHECK( DCheck::check(f4, f2) == false );
    BOOST_CHECK( DCheck::check(f4, f3) == true );
    BOOST_CHECK( DCheck::check(f4, f4) == true );
    BOOST_CHECK( DCheck::check(f4, f5) == true );

    BOOST_CHECK( DCheck::check(f5, f1) == true );
    BOOST_CHECK( DCheck::check(f5, f2) == true );
    BOOST_CHECK( DCheck::check(f5, f3) == true );
    BOOST_CHECK( DCheck::check(f5, f4) == true );
    BOOST_CHECK( DCheck::check(f5, f5) == true );
}

BOOST_AUTO_TEST_SUITE_END();


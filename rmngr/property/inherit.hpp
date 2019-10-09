/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
 * @file rmngr/property/inherit.hpp
 */

#pragma once

namespace rmngr
{

template <
    typename T_Head,
    typename... T_Tail
>
struct TaskPropertiesInherit
    : T_Head
    , TaskPropertiesInherit< T_Tail ... >
{
    template < typename PropertiesBuilder >
    struct Builder
        : T_Head::template Builder< PropertiesBuilder >
        , TaskPropertiesInherit< T_Tail ... >::template Builder< PropertiesBuilder >
    {
        Builder( PropertiesBuilder & p )
            : T_Head::template Builder< PropertiesBuilder >{ p }
            , TaskPropertiesInherit< T_Tail ... >::template Builder< PropertiesBuilder >( p )
        {}
    };

    struct Patch
        : T_Head::Patch
        , TaskPropertiesInherit< T_Tail ... >::Patch
    {
        template < typename PatchBuilder >
        struct Builder
            : T_Head::Patch::template Builder< PatchBuilder >
            , TaskPropertiesInherit< T_Tail ... >::Patch::template Builder< PatchBuilder >
        {
            Builder( PatchBuilder & p )
                : T_Head::Patch::template Builder< PatchBuilder >{ p }
                , TaskPropertiesInherit< T_Tail ... >::Patch::template Builder< PatchBuilder >( p )
            {}
        };
    };

    void apply_patch( Patch const & patch )
    {
        T_Head::apply_patch( patch );
        TaskPropertiesInherit< T_Tail ... >::apply_patch( patch );
    }
};

struct PropEnd_t {};

template<>
struct TaskPropertiesInherit< PropEnd_t >
{
    template < typename PropertiesBuilder >
    struct Builder
    {
        Builder( PropertiesBuilder & ) {}
    };

    struct Patch
    {
        template < typename PatchBuilder >
        struct Builder
        {
            Builder( PatchBuilder & ) {}
        };        
    };
    void apply_patch( Patch const & ) {}
};

template < typename... Policies >
struct TaskProperties
    : public TaskPropertiesInherit< Policies..., PropEnd_t >
{
    struct Builder
        : TaskPropertiesInherit< Policies..., PropEnd_t >::template Builder< Builder >
    {
        TaskProperties prop;

        Builder()
            : TaskPropertiesInherit< Policies..., PropEnd_t >::template Builder< Builder >( *this )
        {}

        Builder( Builder const & b )
            : prop( b.prop )
            , TaskPropertiesInherit< Policies..., PropEnd_t >::template Builder< Builder >( *this )
        {}

        operator TaskProperties () const
        {
            return prop;
        }
    };

    struct Patch
        : TaskPropertiesInherit< Policies..., PropEnd_t >::Patch
    {
        struct Builder
            : TaskPropertiesInherit< Policies..., PropEnd_t >::Patch::template Builder< Builder >
        {
            Patch patch;

            Builder()
                : TaskPropertiesInherit< Policies..., PropEnd_t >::Patch::template Builder< Builder >( *this )
            {}

            Builder( Builder const & b )
                : patch( b.patch )
                , TaskPropertiesInherit< Policies..., PropEnd_t >::Patch::template Builder< Builder >( *this )
            {}

            operator Patch () const
            {
                return patch;
            }
        };
    };

    void apply_patch( Patch const & patch )
    {
        TaskPropertiesInherit< Policies..., PropEnd_t >::apply_patch( patch );
    }
};

}; done

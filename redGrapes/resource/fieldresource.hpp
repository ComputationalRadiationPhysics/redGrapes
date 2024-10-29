/* Copyright 2019-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
 * @file redGrapes/resource/fieldresource.hpp
 */

#pragma once

#include "redGrapes/resource/access/field.hpp"
#include "redGrapes/resource/resource.hpp"
#include "redGrapes/util/traits.hpp"

#include <type_traits>

namespace redGrapes
{

    namespace trait
    {

        template<typename Container>
        struct Field
        {
        };

        template<typename T>
        struct Field<std::vector<T>>
        {
            using Item = T;
            static constexpr size_t dim = 1;

            static std::array<size_t, dim> extent(std::vector<T>& v)
            {
                return {v.size()};
            }

            static Item& get(std::vector<T>& v, std::array<size_t, dim> const& index)
            {
                return v[index[0]];
            }
        };

        template<typename T>
        struct Field<std::vector<T> const>
        {
            using Item = T;
            static constexpr size_t dim = 1;

            static std::array<size_t, dim> extent(std::vector<T>& v)
            {
                return {v.size()};
            }

            static Item const& get(std::vector<T> const& v, std::array<size_t, dim> const& index)
            {
                return v[index[0]];
            }
        };

        template<typename T, size_t N>
        struct Field<std::array<T, N>>
        {
            using Item = T;
            static constexpr size_t dim = 1;

            static Item& get(std::array<T, N>& array, std::array<size_t, dim> const& index)
            {
                return array[index[0]];
            }
        };

        template<typename T, size_t N>
        struct Field<std::array<T, N> const>
        {
            using Item = T;
            static constexpr size_t dim = 1;

            static Item const& get(std::array<T, N> const& array, std::array<size_t, dim> const& index)
            {
                return array[index[0]];
            }
        };

        template<typename T, size_t Nx, size_t Ny>
        struct Field<std::array<std::array<T, Nx>, Ny>>
        {
            using Item = T;
            static constexpr size_t dim = 2;

            static Item& get(std::array<std::array<T, Nx>, Ny>& array, std::array<size_t, dim> const& index)
            {
                return array[index[1]][index[0]];
            }
        };

        template<typename T, size_t Nx, size_t Ny>
        struct Field<std::array<std::array<T, Nx>, Ny> const>
        {
            using Item = T;
            static constexpr size_t dim = 2;

            static Item const& get(
                std::array<std::array<T, Nx>, Ny> const& array,
                std::array<size_t, dim> const& index)
            {
                return array[index[1]][index[0]];
            }
        };


    }; // namespace trait

    namespace fieldaccess
    {

        template<typename Container>
        struct FieldAccessWrapper
        {
            static constexpr size_t dim = trait::Field<Container>::dim;
            using Index = std::array<size_t, dim>;

            FieldAccessWrapper(std::shared_ptr<Container> container) : container(std::move(container)), m_area{}
            {
            }

            FieldAccessWrapper(
                std::shared_ptr<Container> container,
                access::ArrayAccess<access::RangeAccess, dim> area)
                : container(std::move(container))
                , m_area{area}
            {
            }

            auto* operator->() const noexcept
            {
                return container.get();
            }

            auto& operator[](Index const& index) const
            {
                return get(index);
            }

            auto& get(Index const& index) const noexcept
            {
                if(!contains(index))
                    throw std::out_of_range("invalid area access");

                return trait::Field<Container>::get(*container, index);
            }

        private:
            bool contains(Index index) const noexcept
            {
                for(size_t d = 0; d < dim; d++)
                {
                    if(index[d] < m_area[d][0] || index[d] >= m_area[d][1])
                        return false;
                }
                return true;
            }

            std::shared_ptr<Container> container;
            access::ArrayAccess<access::RangeAccess, dim> m_area;
        };

    } // namespace fieldaccess

    namespace fieldresource
    {

        template<typename Container>
        struct AreaGuard : SharedResourceObject<Container, access::FieldAccess<trait::Field<Container>::dim>>
        {
            static constexpr size_t dim = trait::Field<Container>::dim;
            using Item = typename trait::Field<Container>::Item;
            using Index = std::array<size_t, dim>;

            AreaGuard(std::shared_ptr<Container> const& obj)
                : SharedResourceObject<Container, access::FieldAccess<dim>>(obj)
                , m_area{}
            {
            }

            template<typename... Args>
            requires(!(traits::is_specialization_of_v<std::decay_t<traits::first_type_t<Args...>>, AreaGuard>
                       || std::is_same_v<std::decay_t<traits::first_type_t<Args...>>, Container*>) )

            AreaGuard(Args&&... args)
                : SharedResourceObject<Container, access::FieldAccess<dim>>(std::forward<Args>(args)...)
                , m_area{}
            {
            }

            template<typename TContainer>
            AreaGuard(AreaGuard<TContainer> const& res, std::shared_ptr<Container> const& obj)
                : SharedResourceObject<Container, access::FieldAccess<dim>>(res, obj)
                , m_area{}
            {
            }

            template<typename TContainer, typename... Args>
            AreaGuard(AreaGuard<TContainer> const& res, Args&&... args)
                : SharedResourceObject<Container, access::FieldAccess<dim>>(res, std::forward<Args>(args)...)
                , m_area{}
            {
            }

            access::ArrayAccess<access::RangeAccess, dim> make_area(Index begin, Index end) const
            {
                std::array<access::RangeAccess, dim> sub_area;
                for(int d = 0; d < dim; ++d)
                    sub_area[d] = access::RangeAccess({begin[d], end[d]});

                if(!m_area.is_superset_of(sub_area))
                    throw std::out_of_range("invalid sub area");

                return access::ArrayAccess<access::RangeAccess, dim>(sub_area);
            }

        protected:
            access::ArrayAccess<access::RangeAccess, dim> m_area;
        };


    } // namespace fieldresource

    template<typename Container>
    struct FieldResource : fieldresource::AreaGuard<Container>
    {
        static constexpr size_t dim = trait::Field<Container>::dim;
        using typename fieldresource::AreaGuard<Container>::Index;
        using typename fieldresource::AreaGuard<Container>::Item;

        using fieldresource::AreaGuard<Container>::AreaGuard;

        auto access(access::FieldAccess<dim> mode) const noexcept
        {
            return ResourceAccessPair<fieldaccess::FieldAccessWrapper<Container>>{
                this->obj,
                this->res.make_access(mode)};
        }

        auto read() const noexcept
        {
            return ResourceAccessPair<fieldaccess::FieldAccessWrapper<Container const>>{
                this->obj,
                this->res.make_access(access::FieldAccess<dim>(
                    access::IOAccess::read,
                    access::ArrayAccess<access::RangeAccess, dim>{}))};
        }

        auto read(Index begin, Index end) const noexcept
        {
            return ResourceAccessPair<fieldaccess::FieldAccessWrapper<Container const>>{
                this->obj,
                this->res.make_access(access::FieldAccess<dim>(access::IOAccess::read, this->make_area(begin, end)))};
        }

        auto read(Index pos) const noexcept
        {
            Index end = pos;
            for(size_t d = 0; d < dim; ++d)
                end[d]++;
            return ResourceAccessPair<fieldaccess::FieldAccessWrapper<Container const>>{
                {this->obj, this->make_area(pos, end)},
                this->res.make_access(access::FieldAccess<dim>(access::IOAccess::read, this->make_area(pos, end)))};
        }

        auto write() const noexcept
        {
            return ResourceAccessPair<fieldaccess::FieldAccessWrapper<Container>>{
                this->obj,
                this->res.make_access(access::FieldAccess<dim>(
                    access::IOAccess::write,
                    access::ArrayAccess<access::RangeAccess, dim>{}))};
        }

        auto write(Index begin, Index end) const noexcept
        {
            return ResourceAccessPair<fieldaccess::FieldAccessWrapper<Container>>{
                this->obj,
                this->res.make_access(access::FieldAccess<dim>(access::IOAccess::write, this->make_area(begin, end)))};
        }

        auto write(Index pos) const noexcept
        {
            Index end = pos;
            for(size_t d = 0; d < dim; ++d)
                end[d]++;
            return ResourceAccessPair<fieldaccess::FieldAccessWrapper<Container>>{
                this->obj,
                this->res.make_access(access::FieldAccess<dim>(access::IOAccess::write, this->make_area(pos, end)))};
        }
    };

}; // namespace redGrapes

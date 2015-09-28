#include "is_inside.h"

#include <cassert>
#include <list>
#include <limits>
#include <vector>

#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>

#include "order_facets_around_edge.h"
#include "assign_scalar.h"
#include "intersect_other.h"
#include "RemeshSelfIntersectionsParam.h"

namespace igl {
    namespace cgal {
        namespace is_inside_helper {
            typedef CGAL::Exact_predicates_exact_constructions_kernel Kernel;
            typedef Kernel::Ray_3 Ray_3;
            typedef Kernel::Point_3 Point_3;
            typedef Kernel::Vector_3 Vector_3;
            typedef Kernel::Triangle_3 Triangle;
            typedef Kernel::Plane_3 Plane_3;
            typedef std::vector<Triangle>::iterator Iterator;
            typedef CGAL::AABB_triangle_primitive<Kernel, Iterator> Primitive;
            typedef CGAL::AABB_traits<Kernel, Primitive> AABB_triangle_traits;
            typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;

            template<typename DerivedV, typename DerivedF, typename DerivedI>
            bool intersect_each_other(
                    const Eigen::PlainObjectBase<DerivedV>& V1,
                    const Eigen::PlainObjectBase<DerivedF>& F1,
                    const Eigen::PlainObjectBase<DerivedI>& I1,
                    const Eigen::PlainObjectBase<DerivedV>& V2,
                    const Eigen::PlainObjectBase<DerivedF>& F2,
                    const Eigen::PlainObjectBase<DerivedI>& I2) {
                const size_t num_faces_1 = I1.rows();
                DerivedF F1_selected(num_faces_1, F1.cols());
                for (size_t i=0; i<num_faces_1; i++) {
                    F1_selected.row(i) = F1.row(I1(i,0));
                }

                const size_t num_faces_2 = I2.rows();
                DerivedF F2_selected(num_faces_2, F2.cols());
                for (size_t i=0; i<num_faces_2; i++) {
                    F2_selected.row(i) = F2.row(I2(i,0));
                }

                DerivedV VVA, VVB;
                DerivedF IF, FFA, FFB;
                Eigen::VectorXi JA, IMA, JB, IMB;
                RemeshSelfIntersectionsParam param;
                param.detect_only = true;
                param.first_only = true;
                return igl::cgal::intersect_other(
                        V1, F1_selected,
                        V2, F2_selected,
                        param, IF,
                        VVA, FFA, JA, IMA,
                        VVB, FFB, JB, IMB);
            }

            enum ElementType { VERTEX, EDGE, FACE };
            template<typename DerivedV, typename DerivedF, typename DerivedI>
            ElementType determine_element_type(
                    const Eigen::PlainObjectBase<DerivedV>& V,
                    const Eigen::PlainObjectBase<DerivedF>& F,
                    const Eigen::PlainObjectBase<DerivedI>& I,
                    const size_t fid, const Point_3& p,
                    size_t& element_index) {
                const Eigen::Vector3i f = F.row(I(fid, 0));
                const Point_3 p0(V(f[0], 0), V(f[0], 1), V(f[0], 2));
                const Point_3 p1(V(f[1], 0), V(f[1], 1), V(f[1], 2));
                const Point_3 p2(V(f[2], 0), V(f[2], 1), V(f[2], 2));

                if (p == p0) { element_index = 0; return VERTEX; }
                if (p == p1) { element_index = 1; return VERTEX; }
                if (p == p2) { element_index = 2; return VERTEX; }
                if (CGAL::collinear(p0, p1, p)) { element_index = 2; return EDGE; }
                if (CGAL::collinear(p1, p2, p)) { element_index = 0; return EDGE; }
                if (CGAL::collinear(p2, p0, p)) { element_index = 1; return EDGE; }

                element_index = 0;
                return FACE;
            }

            template<typename DerivedV, typename DerivedF, typename DerivedI>
            void extract_adj_faces(
                    const Eigen::PlainObjectBase<DerivedV>& V,
                    const Eigen::PlainObjectBase<DerivedF>& F,
                    const Eigen::PlainObjectBase<DerivedI>& I,
                    const size_t s, const size_t d,
                    std::vector<int>& adj_faces) {
                const size_t num_faces = I.rows();
                for (size_t i=0; i<num_faces; i++) {
                    Eigen::Vector3i f = F.row(I(i, 0));
                    if ((f[0] == s && f[1] == d) ||
                        (f[1] == s && f[2] == d) ||
                        (f[2] == s && f[0] == d)) {
                        adj_faces.push_back((I(i, 0)+1) * -1);
                        continue;
                    }
                    if ((f[0] == d && f[1] == s) ||
                        (f[1] == d && f[2] == s) ||
                        (f[2] == d && f[0] == s)) {
                        adj_faces.push_back(I(i, 0)+1);
                        continue;
                    }
                }
            }

            template<typename DerivedV, typename DerivedF, typename DerivedI>
            void extract_adj_vertices(
                    const Eigen::PlainObjectBase<DerivedV>& V,
                    const Eigen::PlainObjectBase<DerivedF>& F,
                    const Eigen::PlainObjectBase<DerivedI>& I,
                    const size_t v, std::vector<int>& adj_vertices) {
                std::set<size_t> unique_adj_vertices;
                const size_t num_faces = I.rows();
                for (size_t i=0; i<num_faces; i++) {
                    Eigen::Vector3i f = F.row(I(i, 0));
                    assert((f.array() < V.rows()).all());
                    if (f[0] == v) {
                        unique_adj_vertices.insert(f[1]);
                        unique_adj_vertices.insert(f[2]);
                    } else if (f[1] == v) {
                        unique_adj_vertices.insert(f[0]);
                        unique_adj_vertices.insert(f[2]);
                    } else if (f[2] == v) {
                        unique_adj_vertices.insert(f[0]);
                        unique_adj_vertices.insert(f[1]);
                    }
                }
                adj_vertices.resize(unique_adj_vertices.size());
                std::copy(unique_adj_vertices.begin(),
                        unique_adj_vertices.end(),
                        adj_vertices.begin());
            }

            template<typename DerivedV, typename DerivedF, typename DerivedI>
            bool determine_point_edge_orientation(
                    const Eigen::PlainObjectBase<DerivedV>& V,
                    const Eigen::PlainObjectBase<DerivedF>& F,
                    const Eigen::PlainObjectBase<DerivedI>& I,
                    const Point_3& query, size_t s, size_t d) {
                // Algorithm:
                //
                // If the query point is projected onto an edge, all adjacent
                // faces of that edge must be on or belong to a single half
                // space (i.e. there exists a plane passing through the edge and
                // all adjacent faces are either on the plane or on the same
                // side of that plane).
                //
                // If these adjacent faces are not coplanar, query is inside iff
                // the edge is concave.
                //
                // If two or more faces are coplanar, the query point is
                // definitely outside of the 

                std::vector<int> adj_faces;
                extract_adj_faces(V, F, I, s, d, adj_faces);
                const size_t num_adj_faces = adj_faces.size();
                assert(num_adj_faces > 0);
                //std::cout << "adj faces: ";
                //for (size_t i=0; i<num_adj_faces; i++) {
                //    std::cout << adj_faces[i] << " ";
                //}
                //std::cout << std::endl;

                DerivedV pivot_point(1, 3);
                igl::cgal::assign_scalar(query.x(), pivot_point(0, 0));
                igl::cgal::assign_scalar(query.y(), pivot_point(0, 1));
                igl::cgal::assign_scalar(query.z(), pivot_point(0, 2));
                //{
                //    auto get_opposite_vertex = [&](int fid) -> size_t{
                //        Eigen::Vector3i f = F.row(abs(fid)-1);
                //        if (f[0] != s && f[0] != d) return f[0];
                //        if (f[1] != s && f[1] != d) return f[1];
                //        if (f[2] != s && f[2] != d) return f[2];
                //        return -1;
                //    };
                //    Point_3 p_s(V(s,0), V(s,1), V(s,2));
                //    Point_3 p_d(V(d,0), V(d,1), V(d,2));
                //    //std::cout << "s: "
                //    //    << CGAL::to_double(V(s,0)) << " "
                //    //    << CGAL::to_double(V(s,1)) << " "
                //    //    << CGAL::to_double(V(s,2)) << std::endl;
                //    //std::cout << "d: "
                //    //    << CGAL::to_double(V(d,0)) << " "
                //    //    << CGAL::to_double(V(d,1)) << " "
                //    //    << CGAL::to_double(V(d,2)) << std::endl;
                //    for (size_t i=0; i<num_adj_faces; i++) {
                //        size_t o = get_opposite_vertex(adj_faces[i]);
                //        Point_3 p_o(V(o,0), V(o,1), V(o,2));
                //        std::cout << "o" << i << ": "
                //            << CGAL::to_double(V(o,0)) << " "
                //            << CGAL::to_double(V(o,1)) << " "
                //            << CGAL::to_double(V(o,2)) << std::endl;
                //        switch (CGAL::orientation(p_s, p_d, p_o, query)) {
                //            case CGAL::POSITIVE:
                //                std::cout << adj_faces[i] << " positive"  <<
                //                    std::endl;
                //                break;
                //            case CGAL::NEGATIVE:
                //                std::cout << adj_faces[i] << " negative"  <<
                //                    std::endl;
                //                break;
                //            case CGAL::COPLANAR:
                //                std::cout << adj_faces[i] << " coplanar"  <<
                //                    std::endl;
                //                break;
                //            default:
                //                break;
                //        }
                //        //assert(!CGAL::coplanar(p_s, p_d, p_o, query));
                //    }
                //}
                Eigen::VectorXi order;
                order_facets_around_edge(V, F, s, d,
                        adj_faces, pivot_point, order);
                //std::cout << "order: " << order.transpose() << std::endl;
                assert(order.size() == num_adj_faces);
                if (adj_faces[order[0]] > 0 &&
                    adj_faces[order[num_adj_faces-1] < 0]) {
                    return true;
                } else if (adj_faces[order[0]] < 0 &&
                    adj_faces[order[num_adj_faces-1] > 0]) {
                    return false;
                } else {
                    assert(false);
                }
                assert(false);
                return false;
            }

            template<typename DerivedV, typename DerivedF, typename DerivedI>
            bool determine_point_vertex_orientation(
                    const Eigen::PlainObjectBase<DerivedV>& V,
                    const Eigen::PlainObjectBase<DerivedF>& F,
                    const Eigen::PlainObjectBase<DerivedI>& I,
                    const Point_3& query, size_t s) {
                std::vector<int> adj_vertices;
                extract_adj_vertices(V, F, I, s, adj_vertices);
                const size_t num_adj_vertices = adj_vertices.size();

                //std::cout << "Q: "
                //    << CGAL::to_double(query.x()) << " "
                //    << CGAL::to_double(query.y()) << " "
                //    << CGAL::to_double(query.z()) << " "
                //    << std::endl;

                std::vector<Point_3> adj_points;
                for (size_t i=0; i<num_adj_vertices; i++) {
                    const size_t vi = adj_vertices[i];
                    //std::cout << "P: "
                    //    << CGAL::to_double(V(vi,0)) << " "
                    //    << CGAL::to_double(V(vi,1)) << " "
                    //    << CGAL::to_double(V(vi,2)) << " "
                    //    << std::endl;
                    adj_points.emplace_back(V(vi,0), V(vi,1), V(vi,2));
                }

                // A plane is on the exterior if all adj_points lies on or to
                // one side of the plane.
                auto is_on_exterior = [&](const Plane_3& separator) {
                    size_t positive=0;
                    size_t negative=0;
                    size_t coplanar=0;
                    for (const auto& point : adj_points) {
                        switch(separator.oriented_side(point)) {
                            case CGAL::ON_POSITIVE_SIDE:
                                positive++;
                                break;
                            case CGAL::ON_NEGATIVE_SIDE:
                                negative++;
                                break;
                            case CGAL::ON_ORIENTED_BOUNDARY:
                                coplanar++;
                                break;
                            default:
                                assert(false);
                        }
                    }
                    auto query_orientation = separator.oriented_side(query);
                    bool r =
                        (positive == 0 && query_orientation == CGAL::POSITIVE)
                        ||
                        (negative == 0 && query_orientation == CGAL::NEGATIVE);
                    return r;
                };

                size_t d = std::numeric_limits<size_t>::max();
                //std::cout << "P: "
                //    << CGAL::to_double(V(s,0)) << " "
                //    << CGAL::to_double(V(s,1)) << " "
                //    << CGAL::to_double(V(s,2)) << " "
                //    << std::endl;
                Point_3 p(V(s,0), V(s,1), V(s,2));
                for (size_t i=0; i<num_adj_vertices; i++) {
                    const size_t vi = adj_vertices[i];
                    for (size_t j=i+1; j<num_adj_vertices; j++) {
                        const size_t vj = adj_vertices[j];
                        Plane_3 separator(p, adj_points[i], adj_points[j]);
                        assert(!separator.is_degenerate());
                        if (is_on_exterior(separator)) {
                            d = vi;
                            assert(!CGAL::collinear(p, adj_points[i], query));
                            break;
                        }
                    }
                    if (d < V.rows()) break;
                }
                if (d > V.rows()) {
                    // All adj faces are coplanar, use the first edge.
                    d = adj_vertices[0];
                    //std::cout << "all adj faces are coplanar" << std::endl;
                    //return false;
                }
                //std::cout << "s: " << s << "  d: " << d << std::endl;
                return determine_point_edge_orientation(V, F, I, query, s, d);
            }

            template<typename DerivedV, typename DerivedF, typename DerivedI>
            bool determine_point_face_orientation(
                    const Eigen::PlainObjectBase<DerivedV>& V,
                    const Eigen::PlainObjectBase<DerivedF>& F,
                    const Eigen::PlainObjectBase<DerivedI>& I,
                    const Point_3& query, size_t fid) {
                // Algorithm: A point is on the inside of a face if the
                // tetrahedron formed by them is negatively oriented.

                Eigen::Vector3i f = F.row(I(fid, 0));
                const Point_3 v0(V(f[0], 0), V(f[0], 1), V(f[0], 2));
                const Point_3 v1(V(f[1], 0), V(f[1], 1), V(f[1], 2));
                const Point_3 v2(V(f[2], 0), V(f[2], 1), V(f[2], 2));
                auto result = CGAL::orientation(v0, v1, v2, query);
                assert(result != CGAL::COPLANAR);
                return result == CGAL::NEGATIVE;
            }
        }
    }
}

template <typename DerivedV, typename DerivedF, typename DerivedI>
IGL_INLINE bool igl::cgal::is_inside(
        const Eigen::PlainObjectBase<DerivedV>& V1,
        const Eigen::PlainObjectBase<DerivedF>& F1,
        const Eigen::PlainObjectBase<DerivedI>& I1,
        const Eigen::PlainObjectBase<DerivedV>& V2,
        const Eigen::PlainObjectBase<DerivedF>& F2,
        const Eigen::PlainObjectBase<DerivedI>& I2) {
    using namespace igl::cgal::is_inside_helper;
    assert(F1.rows() > 0);
    assert(I1.rows() > 0);
    assert(F2.rows() > 0);
    assert(I2.rows() > 0);

    //assert(!intersect_each_other(V1, F1, I1, V2, F2, I2));

    const size_t num_faces = I2.rows();
    std::vector<Triangle> triangles;
    for (size_t i=0; i<num_faces; i++) {
        const Eigen::Vector3i f = F2.row(I2(i, 0));
        triangles.emplace_back(
                Point_3(V2(f[0], 0), V2(f[0], 1), V2(f[0], 2)),
                Point_3(V2(f[1], 0), V2(f[1], 1), V2(f[1], 2)),
                Point_3(V2(f[2], 0), V2(f[2], 1), V2(f[2], 2)));
        assert(!triangles.back().is_degenerate());
    }
    Tree tree(triangles.begin(), triangles.end());
    tree.accelerate_distance_queries();

    const Eigen::Vector3i& f = F1.row(I1(0, 0));
    const Point_3 query(
            (V1(f[0],0) + V1(f[1],0) + V1(f[2],0))/3.0,
            (V1(f[0],1) + V1(f[1],1) + V1(f[2],1))/3.0,
            (V1(f[0],2) + V1(f[1],2) + V1(f[2],2))/3.0);
    // Computing the closest point to mesh2 is the only exact construction
    // needed in the algorithm.
    auto projection = tree.closest_point_and_primitive(query);
    auto closest_point = projection.first;
    size_t fid = projection.second - triangles.begin();

    size_t element_index;
    switch (determine_element_type(
                V2, F2, I2, fid, closest_point, element_index)) {
        case VERTEX:
            {
                //std::cout << "vertex case" << std::endl;
                const size_t s = F2(I2(fid, 0), element_index);
                return determine_point_vertex_orientation(
                        V2, F2, I2, query, s);
            }
            break;
        case EDGE:
            {
                //std::cout << "edge case" << std::endl;
                const size_t s = F2(I2(fid, 0), (element_index+1)%3);
                const size_t d = F2(I2(fid, 0), (element_index+2)%3);
                return determine_point_edge_orientation(
                        V2, F2, I2, query, s, d);
            }
            break;
        case FACE:
            //std::cout << "face case" << std::endl;
            return determine_point_face_orientation(V2, F2, I2, query, fid);
            break;
        default:
            assert(false);
    }
    assert(false);
    return false;
}

template<typename DerivedV, typename DerivedF>
IGL_INLINE bool igl::cgal::is_inside(
        const Eigen::PlainObjectBase<DerivedV>& V1,
        const Eigen::PlainObjectBase<DerivedF>& F1,
        const Eigen::PlainObjectBase<DerivedV>& V2,
        const Eigen::PlainObjectBase<DerivedF>& F2) {
    Eigen::VectorXi I1(F1.rows()), I2(F2.rows());
    I1.setLinSpaced(F1.rows(), 0, F1.rows()-1);
    I2.setLinSpaced(F2.rows(), 0, F2.rows()-1);
    return igl::cgal::is_inside(V1, F1, I1, V2, F2, I2);
}

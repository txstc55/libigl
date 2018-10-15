#include <npe.h>
#include <typedefs.h>
#include <igl/components.h>

const char* ds_components = R"igl_Qu8mg5v7(
Compute connected components of a graph represented by a sparse adjacency
matrix.

Parameters
----------
a : n by n sparse adjacency matrix

Returns
-------
A tuple (c, counts) where c is an array of component ids (starting with 0)
and counts is a #components array of counts for each component

See also
--------
components_from_faces

Notes
-----

Examples
--------
  
)igl_Qu8mg5v7";

npe_function(components)
npe_doc(ds_components)
npe_arg(a, sparse_i32, sparse_i64)
npe_begin_code()
    EigenDense<npe_Scalar_a> c;
    EigenDense<npe_Scalar_a> counts;
    igl::components(a, c, counts);
    return std::make_tuple(npe::move(c), npe::move(counts));
npe_end_code()



const char* ds_components_from_faces = R"igl_Qu8mg5v7(
Compute connected components of from the face indices of a mesh.

Parameters
----------
f : #f x dim array of face indices

Returns
-------
An array of component ids (starting with 0)

See also
--------
components

Notes
-----

Examples
--------

)igl_Qu8mg5v7";

npe_function(components_from_faces)
npe_doc(ds_components_from_faces)
npe_arg(f, dense_i32, dense_i64)
npe_begin_code()
    npe_Matrix_f c;
    igl::components(f, c);
    return npe::move(c);
npe_end_code()


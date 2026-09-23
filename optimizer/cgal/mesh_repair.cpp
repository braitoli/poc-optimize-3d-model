// mesh_repair — CGAL engine for the pipeline's "face reduction / mesh repair" step.
//
// Contract (see optimizer/cgal/README of the calling Python step):
//   mesh_repair --in <in.ply> --out <out.ply> [--repair] [--self-intersection]
//               [--isolated-min-faces N] [--merge-target-faces N] [--kept-faces <path.bin>]
//
// Operations run in this order, each only when its flag is present:
//   repair -> self-intersection -> isolated -> merge
// Every removal-only op tracks which ORIGINAL face indices survive. Merging destroys
// face identity, so it is mutually exclusive with --kept-faces.
//
// Exactly one line of JSON on stdout on success; on any failure a single
// "Fatal mesh_repair error: <reason>" line on stderr and a non-zero exit code.

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/GarlandHeckbert_plane_policies.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Bounded_normal_change_filter.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Face_count_stop_predicate.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace PMP = CGAL::Polygon_mesh_processing;
namespace SMS = CGAL::Surface_mesh_simplification;

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point3 = Kernel::Point_3;
using Mesh = CGAL::Surface_mesh<Point3>;

// ---------------------------------------------------------------------------
// Fail fast: every error path goes through this.
// ---------------------------------------------------------------------------
[[noreturn]] static void fail(const std::string& reason) { throw std::runtime_error(reason); }

// ---------------------------------------------------------------------------
// Triangle soup. `orig` keeps each row's index in the input file, which is what
// --kept-faces reports; rows are only ever dropped, never reordered.
// ---------------------------------------------------------------------------
struct Soup {
  std::vector<std::array<double, 3>> points;
  std::vector<std::array<int32_t, 3>> tris;
  std::vector<int32_t> orig;
};

// ---------------------------------------------------------------------------
// Minimal binary-little-endian PLY reader.
// Handles arbitrary extra scalar properties (trimesh writes uv as `double s,t`)
// and arbitrary extra elements; only x/y/z and the face index list are used.
// ---------------------------------------------------------------------------
namespace ply {

enum class Type { I8, U8, I16, U16, I32, U32, F32, F64, NONE };

static Type parse_type(const std::string& s) {
  if (s == "char" || s == "int8") return Type::I8;
  if (s == "uchar" || s == "uint8") return Type::U8;
  if (s == "short" || s == "int16") return Type::I16;
  if (s == "ushort" || s == "uint16") return Type::U16;
  if (s == "int" || s == "int32") return Type::I32;
  if (s == "uint" || s == "uint32") return Type::U32;
  if (s == "float" || s == "float32") return Type::F32;
  if (s == "double" || s == "float64") return Type::F64;
  return Type::NONE;
}

static size_t type_size(Type t) {
  switch (t) {
    case Type::I8: case Type::U8: return 1;
    case Type::I16: case Type::U16: return 2;
    case Type::I32: case Type::U32: case Type::F32: return 4;
    case Type::F64: return 8;
    default: return 0;
  }
}

// Reads one scalar as double; integers and floats both go through this since the
// only consumers are coordinates (double) and indices (integral, exactly
// representable for any mesh that fits in int32).
static double read_scalar(const uint8_t* p, Type t) {
  switch (t) {
    case Type::I8:  { int8_t v;   std::memcpy(&v, p, 1); return v; }
    case Type::U8:  { uint8_t v;  std::memcpy(&v, p, 1); return v; }
    case Type::I16: { int16_t v;  std::memcpy(&v, p, 2); return v; }
    case Type::U16: { uint16_t v; std::memcpy(&v, p, 2); return v; }
    case Type::I32: { int32_t v;  std::memcpy(&v, p, 4); return v; }
    case Type::U32: { uint32_t v; std::memcpy(&v, p, 4); return static_cast<double>(v); }
    case Type::F32: { float v;    std::memcpy(&v, p, 4); return v; }
    case Type::F64: { double v;   std::memcpy(&v, p, 8); return v; }
    default: fail("internal: unknown PLY scalar type");
  }
}

struct Prop {
  std::string name;
  bool is_list = false;
  Type count_type = Type::NONE;  // list only
  Type value_type = Type::NONE;
};

struct Element {
  std::string name;
  size_t count = 0;
  std::vector<Prop> props;
};

static Soup read(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) fail("cannot open input file: " + path);
  std::string blob((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (blob.size() < 4 || blob.compare(0, 3, "ply") != 0) fail("not a PLY file: " + path);

  const size_t header_end = blob.find("end_header");
  if (header_end == std::string::npos) fail("PLY header has no end_header: " + path);
  size_t data_start = blob.find('\n', header_end);
  if (data_start == std::string::npos) fail("PLY header is truncated: " + path);
  ++data_start;

  // --- header ---
  std::vector<Element> elements;
  bool format_ok = false;
  std::istringstream hdr(blob.substr(0, header_end));
  std::string line;
  while (std::getline(hdr, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::istringstream ls(line);
    std::string kw;
    ls >> kw;
    if (kw == "format") {
      std::string fmt;
      ls >> fmt;
      if (fmt != "binary_little_endian")
        fail("PLY format must be binary_little_endian, got '" + fmt + "'");
      format_ok = true;
    } else if (kw == "element") {
      Element e;
      ls >> e.name >> e.count;
      elements.push_back(e);
    } else if (kw == "property") {
      if (elements.empty()) fail("PLY 'property' before any 'element'");
      Prop p;
      std::string t;
      ls >> t;
      if (t == "list") {
        p.is_list = true;
        std::string ct, vt;
        ls >> ct >> vt >> p.name;
        p.count_type = parse_type(ct);
        p.value_type = parse_type(vt);
        if (p.count_type == Type::NONE || p.value_type == Type::NONE)
          fail("unsupported PLY list property types in '" + line + "'");
      } else {
        p.value_type = parse_type(t);
        ls >> p.name;
        if (p.value_type == Type::NONE) fail("unsupported PLY property type '" + t + "'");
      }
      elements.back().props.push_back(p);
    }
  }
  if (!format_ok) fail("PLY header has no 'format' line: " + path);

  // --- body ---
  Soup soup;
  const uint8_t* cur = reinterpret_cast<const uint8_t*>(blob.data()) + data_start;
  const uint8_t* end = reinterpret_cast<const uint8_t*>(blob.data()) + blob.size();
  auto need = [&](size_t n) { if (static_cast<size_t>(end - cur) < n) fail("PLY body is truncated: " + path); };

  for (const Element& e : elements) {
    const bool has_list = std::any_of(e.props.begin(), e.props.end(), [](const Prop& p) { return p.is_list; });

    if (!has_list) {
      // Fixed-stride element: compute offsets once, then stride through.
      size_t stride = 0;
      size_t off[3] = {0, 0, 0};
      Type ty[3] = {Type::NONE, Type::NONE, Type::NONE};
      for (const Prop& p : e.props) {
        const char* names = "xyz";
        for (int k = 0; k < 3; ++k)
          if (p.name.size() == 1 && p.name[0] == names[k]) { off[k] = stride; ty[k] = p.value_type; }
        stride += type_size(p.value_type);
      }
      const bool is_vertex = (e.name == "vertex");
      if (is_vertex) {
        for (int k = 0; k < 3; ++k)
          if (ty[k] == Type::NONE) fail("PLY vertex element is missing x/y/z properties");
        soup.points.resize(e.count);
      }
      need(stride * e.count);
      for (size_t i = 0; i < e.count; ++i, cur += stride)
        if (is_vertex)
          for (int k = 0; k < 3; ++k) soup.points[i][k] = read_scalar(cur + off[k], ty[k]);
      continue;
    }

    // Element with a list property: must be walked row by row.
    const bool is_face = (e.name == "face");
    if (is_face) soup.tris.reserve(e.count);
    for (size_t i = 0; i < e.count; ++i) {
      for (const Prop& p : e.props) {
        if (!p.is_list) { need(type_size(p.value_type)); cur += type_size(p.value_type); continue; }
        need(type_size(p.count_type));
        const size_t n = static_cast<size_t>(read_scalar(cur, p.count_type));
        cur += type_size(p.count_type);
        const size_t vsz = type_size(p.value_type);
        need(vsz * n);
        const bool is_indices = is_face && (p.name == "vertex_indices" || p.name == "vertex_index");
        if (is_indices) {
          if (n != 3) fail("input mesh must be triangulated, face " + std::to_string(i) +
                           " has " + std::to_string(n) + " vertices");
          std::array<int32_t, 3> t{};
          for (int k = 0; k < 3; ++k) t[k] = static_cast<int32_t>(read_scalar(cur + vsz * k, p.value_type));
          soup.tris.push_back(t);
        }
        cur += vsz * n;
      }
    }
  }

  if (soup.points.empty()) fail("PLY has no vertices: " + path);
  if (soup.tris.empty()) fail("PLY has no triangular faces: " + path);
  const int32_t nv = static_cast<int32_t>(soup.points.size());
  for (size_t i = 0; i < soup.tris.size(); ++i)
    for (int k = 0; k < 3; ++k)
      if (soup.tris[i][k] < 0 || soup.tris[i][k] >= nv)
        fail("face " + std::to_string(i) + " references vertex index " +
             std::to_string(soup.tris[i][k]) + " out of range");

  soup.orig.resize(soup.tris.size());
  std::iota(soup.orig.begin(), soup.orig.end(), 0);
  return soup;
}

// Writes positions as float32 and triangles as `list uchar int`, which is what
// trimesh reads back fastest.
static void write(const std::string& path,
                  const std::vector<std::array<double, 3>>& pts,
                  const std::vector<std::array<int32_t, 3>>& tris) {
  std::ofstream out(path, std::ios::binary);
  if (!out) fail("cannot open output file for writing: " + path);
  out << "ply\nformat binary_little_endian 1.0\ncomment generated by mesh_repair\n"
      << "element vertex " << pts.size() << "\n"
      << "property float x\nproperty float y\nproperty float z\n"
      << "element face " << tris.size() << "\n"
      << "property list uchar int vertex_indices\nend_header\n";
  std::vector<float> vbuf(pts.size() * 3);
  for (size_t i = 0; i < pts.size(); ++i)
    for (int k = 0; k < 3; ++k) vbuf[i * 3 + k] = static_cast<float>(pts[i][k]);
  out.write(reinterpret_cast<const char*>(vbuf.data()), static_cast<std::streamsize>(vbuf.size() * 4));
  std::vector<uint8_t> fbuf(tris.size() * 13);
  for (size_t i = 0; i < tris.size(); ++i) {
    fbuf[i * 13] = 3;
    std::memcpy(&fbuf[i * 13 + 1], tris[i].data(), 12);
  }
  out.write(reinterpret_cast<const char*>(fbuf.data()), static_cast<std::streamsize>(fbuf.size()));
  out.flush();
  if (!out) fail("failed while writing output file: " + path);
}

}  // namespace ply

// ---------------------------------------------------------------------------
// Soup helpers
// ---------------------------------------------------------------------------

// Drops every row whose `alive` flag is false, preserving order and `orig`.
static void compact(Soup& s, const std::vector<char>& alive) {
  size_t w = 0;
  for (size_t i = 0; i < s.tris.size(); ++i)
    if (alive[i]) { s.tris[w] = s.tris[i]; s.orig[w] = s.orig[i]; ++w; }
  s.tris.resize(w);
  s.orig.resize(w);
}

// Undirected edge key packed into a single 64-bit integer (vertex count < 2^31).
static inline uint64_t edge_key(int32_t a, int32_t b) {
  if (a > b) std::swap(a, b);
  return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b);
}

// Welds soup vertices that occupy the same position and returns, per soup
// vertex, its welded id. glTF splits a vertex at every UV / normal seam, so raw
// PLY indices make one physical shell look like hundreds of separate pieces;
// only the position tells the truth about connectivity.
//
// Positions are quantised onto a grid of `1e-7 * bbox diagonal`, i.e. roughly
// one float32 ulp: seam copies coming out of glTF are bitwise identical and so
// always land in the same cell, and the epsilon only adds slack for meshes that
// went through a lossy round-trip. Two points within epsilon of each other but
// astride a cell boundary are not welded; that is the accepted cost of keeping
// this to a single hash pass.
static std::vector<int32_t> weld_by_position(const std::vector<std::array<double, 3>>& pts) {
  std::array<double, 3> lo{1e300, 1e300, 1e300}, hi{-1e300, -1e300, -1e300};
  for (const auto& p : pts)
    for (int k = 0; k < 3; ++k) { lo[k] = std::min(lo[k], p[k]); hi[k] = std::max(hi[k], p[k]); }
  double diag2 = 0;
  for (int k = 0; k < 3; ++k) diag2 += (hi[k] - lo[k]) * (hi[k] - lo[k]);
  const double eps = 1e-7 * std::sqrt(diag2);
  if (!(eps > 0)) fail("degenerate bounding box: all vertices coincide");

  struct Cell { long long a, b, c; bool operator==(const Cell& o) const { return a == o.a && b == o.b && c == o.c; } };
  struct CellHash {
    size_t operator()(const Cell& k) const {
      size_t h = 1469598103934665603ULL;
      for (long long v : {k.a, k.b, k.c}) h = (h ^ static_cast<size_t>(v)) * 1099511628211ULL;
      return h;
    }
  };

  std::unordered_map<Cell, int32_t, CellHash> cells;
  cells.reserve(pts.size() * 2);
  std::vector<int32_t> weld(pts.size());
  int32_t next = 0;
  for (size_t i = 0; i < pts.size(); ++i) {
    const Cell c{std::llround(pts[i][0] / eps), std::llround(pts[i][1] / eps), std::llround(pts[i][2] / eps)};
    const auto it = cells.emplace(c, next);
    if (it.second) ++next;
    weld[i] = it.first->second;
  }
  return weld;
}

// Faces actually removed, per reason. `non_manifold_vertex` is kept for a
// stable JSON shape but is always 0: non-manifold vertices are split, not cut.
struct Removed {
  size_t degenerate = 0, duplicate = 0, non_manifold_edge = 0, non_manifold_vertex = 0;
  size_t self_intersection = 0, isolated = 0;
};

// Things found and fixed without dropping a face.
struct Detected {
  size_t non_manifold_vertex = 0;  // vertices that had more than one umbrella
};

// Drops triangles with a repeated vertex index or (near-)zero area; returns how
// many went. The threshold scales with the bounding-box diagonal squared, since
// |cross| is O(diag^2), and sits a few orders of magnitude above double
// round-off (~1e-16 relative).
static size_t drop_degenerate(Soup& s) {
  std::array<double, 3> lo{1e300, 1e300, 1e300}, hi{-1e300, -1e300, -1e300};
  for (const auto& p : s.points)
    for (int k = 0; k < 3; ++k) { lo[k] = std::min(lo[k], p[k]); hi[k] = std::max(hi[k], p[k]); }
  double diag2 = 0;
  for (int k = 0; k < 3; ++k) diag2 += (hi[k] - lo[k]) * (hi[k] - lo[k]);
  const double area_eps = 1e-12 * diag2;

  size_t dropped = 0;
  std::vector<char> alive(s.tris.size(), 1);
  for (size_t i = 0; i < s.tris.size(); ++i) {
    const auto& t = s.tris[i];
    if (t[0] == t[1] || t[1] == t[2] || t[0] == t[2]) { alive[i] = 0; ++dropped; continue; }
    const auto& a = s.points[t[0]]; const auto& b = s.points[t[1]]; const auto& c = s.points[t[2]];
    const double ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
    const double vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
    const double cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
    if (std::sqrt(cx * cx + cy * cy + cz * cz) <= area_eps) { alive[i] = 0; ++dropped; }
  }
  compact(s, alive);
  return dropped;
}

// Drops triangles whose sorted vertex triple already occurred; keeps the first.
static size_t drop_duplicate(Soup& s) {
  size_t dropped = 0;
  std::vector<char> alive(s.tris.size(), 1);
  std::unordered_map<uint64_t, std::vector<int32_t>> seen;  // key on two smallest, disambiguate on third
  seen.reserve(s.tris.size() * 2);
  for (size_t i = 0; i < s.tris.size(); ++i) {
    std::array<int32_t, 3> k = s.tris[i];
    std::sort(k.begin(), k.end());
    auto& bucket = seen[edge_key(k[0], k[1])];
    if (std::find(bucket.begin(), bucket.end(), k[2]) != bucket.end()) { alive[i] = 0; ++dropped; }
    else bucket.push_back(k[2]);
  }
  compact(s, alive);
  return dropped;
}

// ---------------------------------------------------------------------------
// repair: degenerate -> duplicate -> non-manifold edges -> non-manifold vertices
// The first three drop faces; the last one only duplicates vertices.
// ---------------------------------------------------------------------------
static void op_repair(Soup& s, Removed& rm, Detected& det) {
  rm.degenerate += drop_degenerate(s);
  rm.duplicate += drop_duplicate(s);

  // --- non-manifold edges: drop ALL faces incident to an edge used by >2 faces ---
  {
    std::unordered_map<uint64_t, uint32_t> deg;
    deg.reserve(s.tris.size() * 3);
    for (const auto& t : s.tris)
      for (int k = 0; k < 3; ++k) ++deg[edge_key(t[k], t[(k + 1) % 3])];
    std::vector<char> alive(s.tris.size(), 1);
    for (size_t i = 0; i < s.tris.size(); ++i) {
      const auto& t = s.tris[i];
      for (int k = 0; k < 3; ++k)
        if (deg[edge_key(t[k], t[(k + 1) % 3])] > 2) { alive[i] = 0; ++rm.non_manifold_edge; break; }
    }
    compact(s, alive);
  }

  // --- non-manifold vertices (umbrella test): SPLIT, never drop ---
  // For each vertex v, link the incident faces that share an edge through v.
  // If that one-ring falls apart into several components ("umbrellas"), v is
  // non-manifold and no halfedge structure can hold it. The standard repair --
  // CGAL's PMP::duplicate_non_manifold_vertices, MeshLab's
  // meshing_repair_non_manifold_vertices -- hands every umbrella past the first
  // its own copy of the vertex. That changes the vertex count and nothing else:
  // no face is dropped, no face is reordered, no original index moves. Cutting
  // these faces away instead would punch visible holes in the surface.
  //
  // PMP's version is not reachable from here: it needs a Surface_mesh, and a
  // Surface_mesh cannot be built from this soup until exactly this defect is
  // gone. So the same operation is done on the soup, where it belongs.
  {
    const size_t nv0 = s.points.size();
    std::vector<std::vector<int32_t>> incident(nv0);
    for (size_t i = 0; i < s.tris.size(); ++i)
      for (int k = 0; k < 3; ++k) incident[s.tris[i][k]].push_back(static_cast<int32_t>(i));

    std::vector<int32_t> parent;
    std::unordered_map<int32_t, int32_t> spoke;  // other endpoint -> first local face seen on it
    std::unordered_map<int32_t, int32_t> umbrella;  // component root -> vertex id it should use

    std::function<int32_t(int32_t)> find = [&](int32_t x) {
      while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
      return x;
    };

    // Only the original vertices need testing: every copy carries exactly one
    // umbrella by construction, so it is manifold the moment it is created.
    for (size_t v = 0; v < nv0; ++v) {
      const std::vector<int32_t>& ring = incident[v];
      if (ring.size() < 2) continue;
      const std::array<double, 3> pv = s.points[v];  // copy: s.points grows below

      parent.resize(ring.size());
      std::iota(parent.begin(), parent.end(), 0);
      spoke.clear();
      for (size_t li = 0; li < ring.size(); ++li) {
        const auto& t = s.tris[ring[li]];
        for (int k = 0; k < 3; ++k) {
          if (t[k] == static_cast<int32_t>(v)) continue;
          auto it = spoke.find(t[k]);
          if (it == spoke.end()) { spoke.emplace(t[k], static_cast<int32_t>(li)); continue; }
          const int32_t ra = find(static_cast<int32_t>(li)), rb = find(it->second);
          if (ra != rb) parent[ra] = rb;
        }
      }

      // First umbrella keeps v; each further one gets a fresh copy of it.
      umbrella.clear();
      for (size_t li = 0; li < ring.size(); ++li) {
        const int32_t root = find(static_cast<int32_t>(li));
        auto it = umbrella.find(root);
        if (it == umbrella.end()) {
          if (umbrella.empty()) { umbrella.emplace(root, static_cast<int32_t>(v)); continue; }
          s.points.push_back(pv);
          it = umbrella.emplace(root, static_cast<int32_t>(s.points.size() - 1)).first;
        }
        auto& t = s.tris[ring[li]];
        for (int k = 0; k < 3; ++k)
          if (t[k] == static_cast<int32_t>(v)) t[k] = it->second;
      }
      if (umbrella.size() > 1) ++det.non_manifold_vertex;
    }
  }
}

// ---------------------------------------------------------------------------
// Soup -> Surface_mesh. Vertices are compacted to the ones actually used and the
// soup is oriented first (CGAL needs a consistently oriented manifold soup);
// orientation never adds or removes polygons, so face i of the mesh is still
// soup row i.
// ---------------------------------------------------------------------------
static Mesh build_mesh(const Soup& s) {
  std::vector<int32_t> remap(s.points.size(), -1);
  std::vector<Point3> pts;
  std::vector<std::vector<size_t>> polys(s.tris.size(), std::vector<size_t>(3));
  for (size_t i = 0; i < s.tris.size(); ++i)
    for (int k = 0; k < 3; ++k) {
      const int32_t v = s.tris[i][k];
      if (remap[v] < 0) {
        remap[v] = static_cast<int32_t>(pts.size());
        pts.emplace_back(s.points[v][0], s.points[v][1], s.points[v][2]);
      }
      polys[i][k] = static_cast<size_t>(remap[v]);
    }

  PMP::orient_polygon_soup(pts, polys);  // may duplicate points; never changes polygon count/order

  Mesh mesh;
  mesh.reserve(pts.size(), pts.size() * 3, polys.size());
  std::vector<Mesh::Vertex_index> vh(pts.size());
  for (size_t i = 0; i < pts.size(); ++i) vh[i] = mesh.add_vertex(pts[i]);
  for (size_t i = 0; i < polys.size(); ++i) {
    const auto f = mesh.add_face(vh[polys[i][0]], vh[polys[i][1]], vh[polys[i][2]]);
    if (f == Mesh::null_face())
      fail("face " + std::to_string(s.orig[i]) + " cannot be added to a manifold surface mesh; "
           "run with --repair first");
  }
  return mesh;
}

// ---------------------------------------------------------------------------
// self-intersection: drop every face reported in an intersecting pair.
// ---------------------------------------------------------------------------
static void op_self_intersection(Soup& s, Removed& rm) {
  const Mesh mesh = build_mesh(s);
  std::vector<std::pair<Mesh::Face_index, Mesh::Face_index>> pairs;
  PMP::self_intersections(mesh, std::back_inserter(pairs));

  std::vector<char> alive(s.tris.size(), 1);
  for (const auto& p : pairs)
    for (const Mesh::Face_index f : {p.first, p.second}) {
      const size_t row = static_cast<size_t>(f);  // faces were added in soup order
      if (row < alive.size() && alive[row]) { alive[row] = 0; ++rm.self_intersection; }
    }
  compact(s, alive);
}

// ---------------------------------------------------------------------------
// isolated: drop connected components with fewer than `min_faces` faces,
// but never the largest component.
//
// PMP::connected_components labels faces over the Surface_mesh's own edges,
// which are raw PLY indices and therefore stop at every UV seam. So the labels
// are only a starting point: two labels that share an edge in POSITION-WELDED
// space are the same physical component and get merged here with a union-find
// over the labels. Welding is purely a connectivity view - no vertex, no face
// and no original index is touched by it.
// ---------------------------------------------------------------------------
static void op_isolated(Soup& s, Removed& rm, size_t min_faces) {
  Mesh mesh = build_mesh(s);
  auto ccmap = mesh.add_property_map<Mesh::Face_index, std::size_t>("f:cc", 0).first;
  const std::size_t n = PMP::connected_components(mesh, ccmap);
  if (n <= 1) return;

  // Union labels joined by a welded edge.
  const std::vector<int32_t> weld = weld_by_position(s.points);
  std::vector<int32_t> parent(n);
  std::iota(parent.begin(), parent.end(), 0);
  std::function<int32_t(int32_t)> find = [&](int32_t x) {
    while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
  };
  std::unordered_map<uint64_t, int32_t> seam;  // welded edge -> first label seen on it
  seam.reserve(s.tris.size() * 3);
  for (size_t i = 0; i < s.tris.size(); ++i) {
    const auto label = static_cast<int32_t>(ccmap[Mesh::Face_index(static_cast<Mesh::size_type>(i))]);
    const auto& t = s.tris[i];
    for (int k = 0; k < 3; ++k) {
      const uint64_t key = edge_key(weld[t[k]], weld[t[(k + 1) % 3]]);
      const auto it = seam.emplace(key, label);
      if (it.second) continue;
      const int32_t ra = find(label), rb = find(it.first->second);
      if (ra != rb) parent[ra] = rb;
    }
  }

  // Face counts per merged component.
  std::unordered_map<int32_t, size_t> sizes;
  std::vector<int32_t> root(n);
  for (size_t c = 0; c < n; ++c) root[c] = find(static_cast<int32_t>(c));
  for (size_t i = 0; i < s.tris.size(); ++i)
    ++sizes[root[ccmap[Mesh::Face_index(static_cast<Mesh::size_type>(i))]]];
  if (sizes.size() < 2) return;

  int32_t largest = sizes.begin()->first;
  size_t largest_n = sizes.begin()->second;
  for (const auto& kv : sizes) if (kv.second > largest_n) { largest = kv.first; largest_n = kv.second; }

  std::vector<char> alive(s.tris.size(), 1);
  for (size_t i = 0; i < s.tris.size(); ++i) {
    const int32_t c = root[ccmap[Mesh::Face_index(static_cast<Mesh::size_type>(i))]];
    if (c == largest || sizes[c] >= min_faces) continue;
    alive[i] = 0;
    ++rm.isolated;
  }
  compact(s, alive);
}

// ---------------------------------------------------------------------------
// Rebuilds the soup with one vertex per distinct position. Welding can turn two
// formerly distinct triangles into the same one, or flatten one to zero area,
// so the same degenerate/duplicate cleanup --repair uses runs on the result:
// both classes carry no geometry and would otherwise make add_face fail.
static Soup weld_soup(const Soup& s) {
  const std::vector<int32_t> weld = weld_by_position(s.points);
  int32_t n = 0;
  for (const int32_t id : weld) n = std::max(n, id + 1);

  Soup w;
  w.points.assign(static_cast<size_t>(n), std::array<double, 3>{0, 0, 0});
  std::vector<char> filled(static_cast<size_t>(n), 0);
  for (size_t i = 0; i < s.points.size(); ++i)
    if (!filled[weld[i]]) { w.points[weld[i]] = s.points[i]; filled[weld[i]] = 1; }

  w.tris.resize(s.tris.size());
  for (size_t i = 0; i < s.tris.size(); ++i)
    for (int k = 0; k < 3; ++k) w.tris[i][k] = weld[s.tris[i][k]];
  w.orig = s.orig;

  drop_degenerate(w);
  drop_duplicate(w);
  return w;
}

// ---------------------------------------------------------------------------
// merge: Garland-Heckbert plane-quadric edge collapse towards `target` faces,
// guarded by Bounded_normal_change_filter so collapses never flip a normal.
//
// The collapse runs on POSITION-WELDED topology. glTF splits a vertex at every
// UV seam, so the raw soup is cut along every seam; edge_collapse would read
// those cuts as mesh borders, refuse to collapse across them, and shred the
// surface into hundreds of loose patches instead of simplifying it. Welding is
// safe here precisely because merging already destroys face identity:
// --kept-faces is rejected in this mode and the caller re-charts UVs anyway.
// ---------------------------------------------------------------------------
static Mesh op_merge(const Soup& s, size_t target) {
  const Soup welded = weld_soup(s);
  Mesh mesh = build_mesh(welded);
  SMS::Face_count_stop_predicate<Mesh> stop(target);
  SMS::GarlandHeckbert_plane_policies<Mesh, Kernel> gh(mesh);
  SMS::Bounded_normal_change_filter<> filter;
  SMS::edge_collapse(mesh, stop,
                     CGAL::parameters::get_cost(gh.get_cost())
                         .get_placement(gh.get_placement())
                         .filter(filter));
  mesh.collect_garbage();
  return mesh;
}

// ---------------------------------------------------------------------------

static std::string json_escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += c; }
    else o += c;
  }
  return o;
}

static size_t parse_count(const std::string& opt, const std::string& v) {
  try {
    const long long n = std::stoll(v);
    if (n < 0) fail(opt + " must be >= 0, got '" + v + "'");
    return static_cast<size_t>(n);
  } catch (const std::invalid_argument&) { fail(opt + " expects an integer, got '" + v + "'"); }
    catch (const std::out_of_range&) { fail(opt + " value out of range: '" + v + "'"); }
}

static int run(int argc, char** argv) {
  std::string in_path, out_path, kept_path;
  bool do_repair = false, do_si = false;
  bool has_isolated = false, has_merge = false;
  size_t isolated_min = 0, merge_target = 0;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&](const char* opt) -> std::string {
      if (i + 1 >= argc) fail(std::string(opt) + " requires a value");
      return argv[++i];
    };
    if (a == "--in") in_path = value("--in");
    else if (a == "--out") out_path = value("--out");
    else if (a == "--kept-faces") kept_path = value("--kept-faces");
    else if (a == "--repair") do_repair = true;
    else if (a == "--self-intersection") do_si = true;
    else if (a == "--isolated-min-faces") { isolated_min = parse_count(a, value("--isolated-min-faces")); has_isolated = true; }
    else if (a == "--merge-target-faces") { merge_target = parse_count(a, value("--merge-target-faces")); has_merge = true; }
    else fail("unknown argument '" + a + "'");
  }

  if (in_path.empty()) fail("--in is required");
  if (out_path.empty()) fail("--out is required");
  if (has_merge && !kept_path.empty())
    fail("--kept-faces cannot be combined with --merge-target-faces: merging destroys face identity");
  if (!has_merge && kept_path.empty())
    fail("--kept-faces is required unless --merge-target-faces is given");
  if (has_merge && merge_target < 1) fail("--merge-target-faces must be >= 1");

  Soup soup = ply::read(in_path);
  const size_t faces_in = soup.tris.size();
  Removed rm;
  Detected det;

  if (do_repair) op_repair(soup, rm, det);
  if (do_si) op_self_intersection(soup, rm);
  if (has_isolated) op_isolated(soup, rm, isolated_min);
  if (soup.tris.empty()) fail("every face was removed; nothing left to write");

  size_t faces_out = soup.tris.size();

  if (has_merge) {
    const Mesh mesh = op_merge(soup, merge_target);
    std::vector<std::array<double, 3>> pts;
    pts.reserve(mesh.number_of_vertices());
    std::unordered_map<size_t, int32_t> vmap;
    vmap.reserve(mesh.number_of_vertices() * 2);
    for (const Mesh::Vertex_index v : mesh.vertices()) {
      const Point3& p = mesh.point(v);
      vmap[static_cast<size_t>(v)] = static_cast<int32_t>(pts.size());
      pts.push_back({p.x(), p.y(), p.z()});
    }
    std::vector<std::array<int32_t, 3>> tris;
    tris.reserve(mesh.number_of_faces());
    for (const Mesh::Face_index f : mesh.faces()) {
      std::array<int32_t, 3> t{};
      int k = 0;
      for (const Mesh::Vertex_index v : CGAL::vertices_around_face(mesh.halfedge(f), mesh)) {
        if (k > 2) fail("simplified mesh contains a non-triangular face");
        t[k++] = vmap[static_cast<size_t>(v)];
      }
      if (k != 3) fail("simplified mesh contains a non-triangular face");
      tris.push_back(t);
    }
    ply::write(out_path, pts, tris);
    faces_out = tris.size();
  } else {
    // Removal-only: compact the vertices the survivors reference and write them out.
    std::vector<int32_t> remap(soup.points.size(), -1);
    std::vector<std::array<double, 3>> pts;
    std::vector<std::array<int32_t, 3>> tris(soup.tris.size());
    for (size_t i = 0; i < soup.tris.size(); ++i)
      for (int k = 0; k < 3; ++k) {
        const int32_t v = soup.tris[i][k];
        if (remap[v] < 0) { remap[v] = static_cast<int32_t>(pts.size()); pts.push_back(soup.points[v]); }
        tris[i][k] = remap[v];
      }
    ply::write(out_path, pts, tris);

    std::ofstream kf(kept_path, std::ios::binary);
    if (!kf) fail("cannot open kept-faces file for writing: " + kept_path);
    kf.write(reinterpret_cast<const char*>(soup.orig.data()),
             static_cast<std::streamsize>(soup.orig.size() * sizeof(int32_t)));
    kf.flush();
    if (!kf) fail("failed while writing kept-faces file: " + kept_path);
  }

  std::ostringstream js;
  js << "{\"facesIn\":" << faces_in << ",\"facesOut\":" << faces_out
     << ",\"removed\":{\"degenerate\":" << rm.degenerate
     << ",\"duplicate\":" << rm.duplicate
     << ",\"nonManifoldEdge\":" << rm.non_manifold_edge
     << ",\"nonManifoldVertex\":" << rm.non_manifold_vertex
     << ",\"selfIntersection\":" << rm.self_intersection
     << ",\"isolated\":" << rm.isolated << "}"
     << ",\"detected\":{\"nonManifoldVertex\":" << det.non_manifold_vertex << "}"
     << ",\"merged\":";
  if (has_merge) js << "{\"requested\":" << merge_target << ",\"facesOut\":" << faces_out << "}";
  else js << "null";
  js << ",\"keptFaces\":";
  if (has_merge) js << "null";
  else js << "{\"file\":\"" << json_escape(kept_path) << "\",\"count\":" << soup.orig.size() << "}";
  js << "}";
  std::cout << js.str() << std::endl;
  return 0;
}

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "Fatal mesh_repair error: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Fatal mesh_repair error: unknown exception" << std::endl;
    return 1;
  }
}

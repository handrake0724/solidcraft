// The curvature map at every pixel of a planned body: the surface's own curvature at the
// fragment's (u, v), not the vertices' curvature interpolated across a triangle. Where the
// curvature changes along a nearly flat stretch (a ship's bilge running into its flat side) the
// display mesh leaves triangles metres long, and two faces meeting at a seam showed different
// colours either side of it. Appended to the viewport shader where the device evaluates plans
// (the surfaces are the evaluation's own buffers, bound in group 3).
//
// The evaluation follows `tess.wgsl` (The NURBS Book A2.1 span search, A2.2 basis functions,
// A2.3 derivatives) with the second derivatives added, in f32; the curvature follows
// `EvalSurface::curvature` (the largest principal curvature in magnitude).

struct ExSurface {
    kind: u32,
    pu: u32,
    pv: u32,
    nu: u32,
    nv: u32,
    knots_u: u32,
    knots_v: u32,
    ctrl: u32,
    extra: u32,
    swap: u32,
    mirror: u32,
    closed: u32,
};

@group(3) @binding(0) var<storage, read> ex_surfaces: array<ExSurface>;
@group(3) @binding(1) var<storage, read> ex_data: array<f32>;

// Basis slots for the highest degree (MAX_SPLINE_DEGREE = 16), as in `tess.wgsl`.
const EX_W: u32 = 17u;
const EX_PLANE: u32 = 0u;
const EX_NURBS: u32 = 1u;

struct XOut {
    @invariant @builtin(position) pos: vec4<f32>,
    @location(0) n: vec3<f32>,
    @location(1) c: vec4<f32>,
    @location(2) wp: vec3<f32>,
    @location(3) @interpolate(flat) h: u32,
    @location(4) k: f32,
    // The surface parameters, and the surface (an index into `ex_surfaces`).
    @location(5) q: vec2<f32>,
    @location(6) @interpolate(flat) s: u32,
};

/// `vs_tri` for planned bodies, with each vertex's parameters and surface (the evaluation's
/// input vertices, read as a third vertex buffer).
@vertex
fn vs_tri_exact(
    @location(0) local: vec3<f32>,
    @location(1) n: vec3<f32>,
    @location(2) c: u32,
    @location(7) k: f32,
    @location(8) q: vec2<f32>,
    @location(9) s: u32,
) -> XOut {
    var o: XOut;
    let p = placed(local);
    o.pos = u.vp * vec4<f32>(p, 1.0);
    o.n = turned(n);
    o.k = k;
    let e = palette_entry(c);
    o.c = unpack_color(e.x);
    o.h = e.y;
    o.wp = p;
    o.q = q;
    o.s = s;
    return o;
}

fn ex_plain(i: XOut) -> TOut {
    var t: TOut;
    t.pos = i.pos;
    t.n = i.n;
    t.c = i.c;
    t.wp = i.wp;
    t.h = i.h;
    t.k = i.k;
    return t;
}

/// The curvature to show: the surface's own at the fragment's parameters while the curvature
/// map is on, else (and where that is not known: a pole, a surface of revolution) as `solid`
/// takes it. The screen-space fallback is worked out first, in uniform control flow.
fn ex_shown(i: XOut, t: TOut) -> f32 {
    var k = curvature(t);
    if (u.ana.x > 2.5 && u.ana.x < 3.5) {
        let e = ex_curvature(i.q, i.s);
        if (e >= 0.0) {
            k = e;
        }
    }
    return k;
}

@fragment
fn fs_solid_exact(i: XOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let t = ex_plain(i);
    return solid_k(t, front, ex_shown(i, t));
}

fn ex_span(k0: u32, p: u32, n: u32, t: f32) -> u32 {
    if (t >= ex_data[k0 + n]) {
        var k = n - 1u;
        while (k > p && ex_data[k0 + k] >= ex_data[k0 + k + 1u]) {
            k = k - 1u;
        }
        return k;
    }
    if (t < ex_data[k0 + p]) {
        return p;
    }
    var lo = p;
    var hi = n;
    while (hi - lo > 1u) {
        let mid = (lo + hi) / 2u;
        if (ex_data[k0 + mid] <= t) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// The non-zero basis functions of degree p on `span` (A2.2).
fn ex_basis(k0: u32, span: u32, p: u32, t: f32) -> array<f32, EX_W> {
    var n: array<f32, EX_W>;
    n[0] = 1.0;
    for (var j = 1u; j <= p; j = j + 1u) {
        var saved = 0.0;
        for (var r = 0u; r < j; r = r + 1u) {
            let right = ex_data[k0 + span + r + 1u] - t;
            let left = t - ex_data[k0 + span + 1u + r - j];
            let den = right + left;
            var temp = 0.0;
            if (den != 0.0) {
                temp = n[r] / den;
            }
            n[r] = saved + right * temp;
            saved = left * temp;
        }
        n[j] = saved;
    }
    return n;
}

// The derivatives of the degree p functions on `span` from `lower`, the same derivative of the
// degree p - 1 functions there: N'_{i,p} = p (N_{i,p-1} / (u_{i+p} - u_i) - N_{i+1,p-1} /
// (u_{i+p+1} - u_{i+1})), applied once for the first derivatives and twice for the second.
fn ex_raise(k0: u32, span: u32, p: u32, lower: array<f32, EX_W>) -> array<f32, EX_W> {
    var d: array<f32, EX_W>;
    if (p == 0u) {
        return d;
    }
    for (var r = 0u; r <= p; r = r + 1u) {
        var x = 0.0;
        if (r >= 1u) {
            let den = ex_data[k0 + span + r] - ex_data[k0 + span + r - p];
            if (den != 0.0) {
                x = x + lower[r - 1u] / den;
            }
        }
        if (r < p) {
            let den = ex_data[k0 + span + r + 1u] - ex_data[k0 + span + r + 1u - p];
            if (den != 0.0) {
                x = x - lower[r] / den;
            }
        }
        d[r] = x * f32(p);
    }
    return d;
}

fn ex_first(k0: u32, span: u32, p: u32, t: f32) -> array<f32, EX_W> {
    var d: array<f32, EX_W>;
    if (p == 0u) {
        return d;
    }
    return ex_raise(k0, span, p, ex_basis(k0, span, p - 1u, t));
}

fn ex_second(k0: u32, span: u32, p: u32, t: f32) -> array<f32, EX_W> {
    var d: array<f32, EX_W>;
    if (p < 2u) {
        return d;
    }
    return ex_raise(k0, span, p, ex_raise(k0, span, p - 1u, ex_basis(k0, span, p - 2u, t)));
}

fn ex_wrap(knots: u32, p: u32, n: u32, t: f32, closed: bool) -> f32 {
    if (!closed) { return t; }
    let a = ex_data[knots + p];
    let b = ex_data[knots + n];
    if (!(b > a) || (t >= a && t <= b)) { return t; }
    return a + (t - a) - floor((t - a) / (b - a)) * (b - a);
}

fn ex_ctrl(o: u32) -> vec4<f32> {
    return vec4<f32>(ex_data[o], ex_data[o + 1u], ex_data[o + 2u], ex_data[o + 3u]);
}

/// The largest principal curvature in magnitude (1/mm) of surface `s` at `q`, from its first
/// and second fundamental forms; 0 on a plane or a flat spline; negative where it is not worked
/// out here (no surface, a surface of revolution, a singular point).
fn ex_curvature(q: vec2<f32>, si: u32) -> f32 {
    if (si >= arrayLength(&ex_surfaces)) {
        return -1.0;
    }
    let s = ex_surfaces[si];
    if (s.kind == EX_PLANE || (s.kind == EX_NURBS && (s.closed & 4u) != 0u)) {
        return 0.0;
    }
    if (s.kind != EX_NURBS) {
        return -1.0;
    }
    let u0 = ex_wrap(s.knots_u, s.pu, s.nu, q.x, (s.closed & 1u) != 0u);
    let v0 = ex_wrap(s.knots_v, s.pv, s.nv, q.y, (s.closed & 2u) != 0u);
    let su = ex_span(s.knots_u, s.pu, s.nu, u0);
    let sv = ex_span(s.knots_v, s.pv, s.nv, v0);
    let bu0 = ex_basis(s.knots_u, su, s.pu, u0);
    let bu1 = ex_first(s.knots_u, su, s.pu, u0);
    let bu2 = ex_second(s.knots_u, su, s.pu, u0);
    let bv0 = ex_basis(s.knots_v, sv, s.pv, v0);
    let bv1 = ex_first(s.knots_v, sv, s.pv, v0);
    let bv2 = ex_second(s.knots_v, sv, s.pv, v0);
    // Control points relative to the first of the window: the derivatives are the same, and a
    // part far from the origin keeps its digits (the basis' second derivatives sum to zero only
    // up to rounding, which would multiply the distance from the origin).
    let first = ex_ctrl(s.ctrl + 4u * ((sv - s.pv) * s.nu + su - s.pu));
    if (!(abs(first.w) > 0.0)) {
        return -1.0;
    }
    let o = first.xyz / first.w;
    var a = vec4<f32>(0.0);
    var au = vec4<f32>(0.0);
    var av = vec4<f32>(0.0);
    var auu = vec4<f32>(0.0);
    var auv = vec4<f32>(0.0);
    var avv = vec4<f32>(0.0);
    for (var j = 0u; j <= s.pv; j = j + 1u) {
        for (var i = 0u; i <= s.pu; i = i + 1u) {
            let h = ex_ctrl(s.ctrl + 4u * ((sv + j - s.pv) * s.nu + su + i - s.pu));
            let c = vec4<f32>(h.xyz - o * h.w, h.w);
            a = a + bu0[i] * bv0[j] * c;
            au = au + bu1[i] * bv0[j] * c;
            av = av + bu0[i] * bv1[j] * c;
            auu = auu + bu2[i] * bv0[j] * c;
            auv = auv + bu1[i] * bv1[j] * c;
            avv = avv + bu0[i] * bv2[j] * c;
        }
    }
    if (!(abs(a.w) > 0.0)) {
        return -1.0;
    }
    // The rational surface's derivatives from the homogeneous ones (quotient rule).
    let w = a.w;
    let p = a.xyz / w;
    let pu = (au.xyz - au.w * p) / w;
    let pv = (av.xyz - av.w * p) / w;
    let puu = (auu.xyz - 2.0 * au.w * pu - auu.w * p) / w;
    let puv = (auv.xyz - au.w * pv - av.w * pu - auv.w * p) / w;
    let pvv = (avv.xyz - 2.0 * av.w * pv - avv.w * p) / w;
    let e = dot(pu, pu);
    let f = dot(pu, pv);
    let g = dot(pv, pv);
    let det = e * g - f * f;
    // A singular point (a pole, a collapsed side): no tangent plane to bend from (f32 keeps
    // about seven digits of the products).
    if (!(det > 1e-5 * e * g)) {
        return -1.0;
    }
    let n = normalize(cross(pu, pv));
    let l = dot(puu, n);
    let m = dot(puv, n);
    let nn = dot(pvv, n);
    let gauss = (l * nn - m * m) / det;
    let mean = (e * nn - 2.0 * f * m + g * l) / (2.0 * det);
    let k = abs(mean) + sqrt(max(mean * mean - gauss, 0.0));
    if (!(k >= 0.0) || k > 3.0e38) {
        return -1.0;
    }
    return k;
}

//! SMLib's own pcurves for a face's uses that have no closed form (an approximated intersection
//! curve on a cone, a file's spline edge on a spline surface): the face goes to SMLib alone, an
//! open sheet, with the trims it has and without the others, and SMLib makes those as it makes
//! every trim it needs (`SmEdgeuse::GetOrCreateUVTrimCurve`, when the body is exported). They are
//! SMLib's, relayed as it holds them on the record's parameters. The one curve fitted here is a
//! pcurve longer than the record holds ([`within_record_size`]).

use solidcraft_geom::prelude::*;
use std::collections::HashSet;

use solidcore_ops::pcurve::{SURFACE_GAP_MARGIN, TrimFace, Trims};
use solidcore_record::brep::{self as b, Curve2, EdgeUseRef, Loop, OrientedEdge, OrientedFace, Region, Shell, Vertex};

/// [`Trims`] by SMLib.
#[derive(Clone, Copy, Debug, Default)]
pub struct SmlibTrims;

impl Trims for SmlibTrims {
    fn trims(&self, face: &TrimFace<'_>) -> Result<Vec<Vec<Curve2>>, String> {
        trims(face).map_err(|e| format!("SMLib pcurves: {e}"))
    }
}

fn trims(face: &TrimFace<'_>) -> Result<Vec<Vec<Curve2>>, String> {
    let (record, bare) = sheet(face)?;
    let mut arrays = crate::import::arrays_with_bare_uses(&record, &bare).map_err(|e| e.to_string())?;
    let import = |a: &crate::import::Arrays| crate::Brep::from_arrays(a).map_err(|e| e.to_string());
    let mut body = import(&arrays)?;
    if cover_gaps(&body, face, &bare, &mut arrays)? {
        body = import(&arrays)?;
    }
    let (out, order) = body.to_native_ordered().map_err(|e| e.to_string())?;
    let [made] = out.faces.as_slice() else { return Err(format!("SMLib made {} faces of one", out.faces.len())) };
    let made_uses: Vec<&OrientedEdge> = made
        .loops
        .iter()
        .flat_map(|l| match l {
            Loop::Edges { edges } => edges.as_slice(),
            Loop::Vertex { .. } => &[],
        })
        .collect();
    // SMLib keeps the edges in the order they came; the record renumbers them.
    let edge_of = |u: u32| order.edges.get(u as usize).copied().ok_or("an edge SMLib lost");
    // A sheet's loops come back the way SMLib runs them, which may be against ours: a use is the
    // same pcurve either way (on its edge's interval in its curve's direction), and the loop's
    // way round is read from an edge the face uses once.
    let ours = face.loops.iter().flatten();
    let once = ours.clone().find(|u| ours.clone().filter(|v| v.edge == u.edge).count() == 1).ok_or("a face of seam edges only")?;
    let at = edge_of(once.edge)?;
    let reversed = made_uses.iter().find(|m| m.edge as usize == at).ok_or("a use SMLib did not keep")?.forward != once.forward;
    // Each edge's tolerance as SMLib held it (raised where it lies off the surface).
    let held = arrays.reals.get(&crate::flat::EDGE_TOLERANCE).ok_or("edge tolerances")?;
    let mut result = Vec::with_capacity(face.loops.len());
    for l in &face.loops {
        let mut pcs = Vec::with_capacity(l.len());
        for u in l {
            let e = edge_of(u.edge)?;
            let edge = out.edges.get(e).ok_or("an edge SMLib lost")?;
            let ours = face.edges.get(u.edge as usize).ok_or("edge reference")?;
            let found = made_uses.iter().find(|m| m.edge as usize == e && m.forward == (u.forward != reversed));
            let pc = found.ok_or("a use SMLib did not keep")?.pcurve.clone();
            let tolerance = held.get(u.edge as usize).copied().unwrap_or(ours.tolerance).max(ours.tolerance);
            pcs.push(on_our_interval(pc, ours, edge.interval, tolerance)?);
        }
        result.push(pcs);
    }
    Ok(result)
}

/// Each bare use's edge measured against the face's surface as SMLib measures it (over the
/// surface's own domain: an edge run a few um past its patch measures how far), and its
/// `EDGE_TOLERANCE` in `arrays` raised to cover that gap with [`SURFACE_GAP_MARGIN`]: SMLib keeps
/// an edge whole when making its trim only when its zone tolerance (half that) covers the gap, and
/// otherwise trims the edge back to the patch. Whether any was raised; an edge farther off than
/// SMLib's tolerances reach is refused with how far.
fn cover_gaps(body: &crate::Brep, face: &TrimFace<'_>, bare: &HashSet<(u32, u32, u32)>, arrays: &mut crate::import::Arrays) -> Result<bool, String> {
    let mut edges: Vec<usize> = Vec::new();
    for (li, l) in face.loops.iter().enumerate() {
        for (p, u) in l.iter().enumerate() {
            let (Ok(li), Ok(p)) = (u32::try_from(li), u32::try_from(p)) else { return Err("too many uses".into()) };
            if bare.contains(&(0, li, p)) && !edges.contains(&(u.edge as usize)) {
                edges.push(u.edge as usize);
            }
        }
    }
    let mut raised = false;
    for e in edges {
        let gap = body.edge_face_distance(e, 0).map_err(|e| e.to_string())?;
        let need = SURFACE_GAP_MARGIN * gap;
        let tolerances = arrays.reals.get_mut(&crate::flat::EDGE_TOLERANCE).ok_or("edge tolerances")?;
        let given = tolerances.get_mut(e).ok_or("edge reference")?;
        if need <= *given {
            continue;
        }
        if !(need <= crate::flat::MAX_TOLERANCE) {
            return Err(format!("an edge lies {gap} mm off the face's surface"));
        }
        *given = need;
        raised = true;
    }
    Ok(raised)
}

/// A pcurve SMLib made on its edge's interval `theirs`, on ours. SMLib may set an edge's end to
/// a parameter of its own (a knot its vertex lies at, the end of the part on the face: where an
/// edge runs past its face's patch, SMLib ends it where its drop onto the patch ends, when that
/// is within the edge's cross-section tolerance of its vertex): where each end it chose is
/// within `tolerance` of ours on the edge's curve (the edge's tolerance as SMLib held it,
/// [`cover_gaps`]), the pcurve is mapped onto our interval by the affine change of parameter
/// (exact for a B-spline: its knots move), and the record's check measures it against the edge
/// as any pcurve. Farther is SMLib trimming the edge, refused with how far.
pub(crate) fn on_our_interval(pc: Curve2, ours: &solidcore_ops::pcurve::TrimEdge<'_>, theirs: [f64; 2], tolerance: f64) -> Result<Curve2, String> {
    let [a, b] = ours.interval;
    let [c, d] = theirs;
    if a == c && b == d {
        return Ok(pc);
    }
    let curve = solidcraft_geom::brep::eval::CurveEval::new(ours.nurbs.unwrap_or(ours.curve)).map_err(|e| e.to_string())?;
    let at = |t: f64| curve.point(t).ok_or_else(|| format!("the edge's curve has no point at {t}"));
    let shift = at(a)?.dist(at(c)?).max(at(b)?.dist(at(d)?));
    if !(shift <= tolerance) || !(d > c) {
        return Err(format!(
            "SMLib moved an edge's interval from {:?} to {:?}, its ends {shift} mm along its curve (tolerance {tolerance} mm)",
            ours.interval, theirs
        ));
    }
    let map = |k: f64| {
        if k == c {
            a
        } else if k == d {
            b
        } else {
            a + (k - c) * ((b - a) / (d - c))
        }
    };
    Ok(match pc {
        Curve2::Spline { knots, points } => Curve2::Spline { knots: knots.into_iter().map(map).collect(), points },
        Curve2::RationalSpline { knots, homogeneous } => Curve2::RationalSpline { knots: knots.into_iter().map(map).collect(), homogeneous },
    })
}

/// The most control points a record's spline holds (`solidcore_record`'s check).
const RECORD_MAX_POINTS: usize = 4096;
/// Samples per span of a refitted pcurve, for the least-squares fit (the measure takes twice as
/// many).
const REFIT_SAMPLES: usize = 6;

/// `pc` as the record holds it. SMLib makes a long trim on a free-form surface as a chain of
/// cubic pieces joined end to end (each interior knot three times), and a propeller blade's edge
/// came to 1538 of them, 4612 control points, past the record's 4096: the face was left out.
/// Such a pcurve is refitted by least squares as a cubic B-spline on the same parameters and
/// the same breaks, each a double knot instead of a triple one (a single knot where that is
/// still too many points, else breaks spread evenly). Its ends stay exact, since they meet the
/// loop's next and previous uses, and it is kept only where it lies within `tolerance` of the
/// original on the face's surface.
pub(crate) fn within_record_size(pc: Curve2, surface: &b::Surface, tolerance: f64) -> Result<Curve2, String> {
    let Curve2::Spline { points, knots: given } = &pc else { return Ok(pc) };
    if points.len() <= RECORD_MAX_POINTS {
        return Ok(pc);
    }
    let [a, b] = pc.interval().ok_or("pcurve without an interval")?;
    let at = |t: f64| solidcore_ops::pcurve::eval(&pc, t).ok_or_else(|| format!("pcurve has no point at {t}"));
    let p = 3usize;
    let mut breaks: Vec<f64> = given.iter().copied().filter(|k| *k > a && *k < b).collect();
    breaks.dedup();
    let interior: Vec<f64> = if 2 * breaks.len() + p < RECORD_MAX_POINTS {
        breaks.iter().flat_map(|k| [*k, *k]).collect()
    } else if breaks.len() + p < RECORD_MAX_POINTS {
        breaks
    } else {
        let spans = RECORD_MAX_POINTS - p - 1;
        (1..spans).map(|i| a + (b - a) * (i as f64 / spans as f64)).collect()
    };
    let mut knots = vec![a; p + 1];
    knots.extend(interior);
    knots.extend(std::iter::repeat_n(b, p + 1));
    let n = knots.len() - p - 1;
    let (first, last) = (at(a)?, at(b)?);
    // The parameters sampled: evenly within each span.
    let mut cuts: Vec<f64> = knots.clone();
    cuts.dedup();
    let sample = |per: usize| -> Vec<f64> {
        let mut ts: Vec<f64> = cuts.windows(2).flat_map(|w| (0..per).map(move |i| w[0] + (w[1] - w[0]) * (i as f64 / per as f64))).collect();
        ts.push(b);
        ts
    };
    // Normal equations for the interior points 1..n-1, banded (half-width p).
    let m = n - 2;
    let mut band = vec![[0.0f64; 4]; m];
    let mut rhs = vec![[0.0f64; 2]; m];
    for t in sample(REFIT_SAMPLES) {
        let q = at(t)?;
        let (span, basis) = bspline_basis(&knots, n, p, t).ok_or("refit basis")?;
        // Fixed ends move to the right-hand side.
        let mut r = q;
        for (k, nk) in basis.iter().enumerate() {
            let i = span + k - p;
            let fixed = if i == 0 {
                Some(first)
            } else if i == n - 1 {
                Some(last)
            } else {
                None
            };
            if let Some(f) = fixed {
                r = [r[0] - nk * f[0], r[1] - nk * f[1]];
            }
        }
        for (k, nk) in basis.iter().enumerate() {
            let i = span + k - p;
            if i == 0 || i == n - 1 {
                continue;
            }
            let row = i - 1;
            if let Some(slot) = rhs.get_mut(row) {
                *slot = [slot[0] + nk * r[0], slot[1] + nk * r[1]];
            }
            for (l, nl) in basis.iter().enumerate().take(k + 1) {
                let j = span + l - p;
                if j == 0 || j == n - 1 {
                    continue;
                }
                // band[row][d]: the entry (row, row - d).
                if let Some(cell) = band.get_mut(row).and_then(|c| c.get_mut(i - j)) {
                    *cell += nk * nl;
                }
            }
        }
    }
    let interior = banded_solve(&mut band, rhs).ok_or("refit system")?;
    let mut out = Vec::with_capacity(n);
    out.push(first);
    out.extend(interior);
    out.push(last);
    let fit = Curve2::Spline { knots, points: out };
    // Measured on the surface, between the samples as well as at them.
    let surface = solidcraft_geom::brep::eval::SurfaceEval::new(surface).map_err(|e| e.to_string())?;
    let mut worst = 0.0f64;
    for t in sample(2 * REFIT_SAMPLES) {
        let (Some(p0), Some(p1)) = (surface.point(at(t)?), solidcore_ops::pcurve::eval(&fit, t).and_then(|uv| surface.point(uv))) else {
            return Err("refitted pcurve off its surface".into());
        };
        worst = worst.max(p0.dist(p1));
    }
    if !(worst <= tolerance) {
        return Err(format!("a pcurve of {} points refitted to {n} strays {worst} mm (tolerance {tolerance} mm)", points.len()));
    }
    Ok(fit)
}

/// The knot span holding `t` and the `p + 1` basis functions nonzero there, of a clamped spline
/// of `n` points.
fn bspline_basis(knots: &[f64], n: usize, p: usize, t: f64) -> Option<(usize, Vec<f64>)> {
    let span = if t >= *knots.get(n)? {
        n - 1
    } else {
        // The last span s in p..n with knots[s] <= t.
        let (mut lo, mut hi) = (p, n);
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if t < *knots.get(mid)? {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        lo
    };
    let mut nb = vec![0.0; p + 1];
    let (mut left, mut right) = (vec![0.0; p + 1], vec![0.0; p + 1]);
    *nb.first_mut()? = 1.0;
    for j in 1..=p {
        *left.get_mut(j)? = t - knots.get(span + 1 - j)?;
        *right.get_mut(j)? = knots.get(span + j)? - t;
        let mut saved = 0.0;
        for r in 0..j {
            let denom = right.get(r + 1)? + left.get(j - r)?;
            let temp = if denom != 0.0 { nb.get(r)? / denom } else { 0.0 };
            *nb.get_mut(r)? = saved + right.get(r + 1)? * temp;
            saved = left.get(j - r)? * temp;
        }
        *nb.get_mut(j)? = saved;
    }
    Some((span, nb))
}

/// Solve a symmetric positive definite banded system (`band[i][d]` the entry (i, i - d), half
/// width 3) for two right-hand sides, by Cholesky in place.
fn banded_solve(band: &mut [[f64; 4]], mut rhs: Vec<[f64; 2]>) -> Option<Vec<[f64; 2]>> {
    let m = band.len();
    for i in 0..m {
        for d in (0..4).rev() {
            if d > i {
                continue;
            }
            let j = i - d;
            // sum over k < j within the band of L(i,k) L(j,k)
            let mut sum = band.get(i)?.get(d).copied()?;
            for k in j.saturating_sub(3)..j {
                if i - k > 3 {
                    continue;
                }
                sum -= band.get(i)?.get(i - k)? * band.get(j)?.get(j - k)?;
            }
            if d == 0 {
                if !(sum > 0.0) {
                    return None;
                }
                *band.get_mut(i)?.get_mut(0)? = sum.sqrt();
            } else {
                *band.get_mut(i)?.get_mut(d)? = sum / band.get(j)?.first()?;
            }
        }
    }
    // L y = r, then Lᵀ x = y.
    for i in 0..m {
        let mut y = *rhs.get(i)?;
        for d in 1..4.min(i + 1) {
            let (l, prev) = (band.get(i)?.get(d)?, rhs.get(i - d)?);
            y = [y[0] - l * prev[0], y[1] - l * prev[1]];
        }
        let diag = band.get(i)?.first()?;
        *rhs.get_mut(i)? = [y[0] / diag, y[1] / diag];
    }
    for i in (0..m).rev() {
        let mut x = *rhs.get(i)?;
        for d in 1..4 {
            if i + d >= m {
                break;
            }
            let (l, next) = (band.get(i + d)?.get(d)?, rhs.get(i + d)?);
            x = [x[0] - l * next[0], x[1] - l * next[1]];
        }
        let diag = band.get(i)?.first()?;
        *rhs.get_mut(i)? = [x[0] / diag, x[1] / diag];
    }
    Some(rhs)
}

/// The face as a one-face open sheet, and its uses that go without trims.
fn sheet(face: &TrimFace<'_>) -> Result<(b::Brep, HashSet<(u32, u32, u32)>), String> {
    let mut edges: Vec<b::Edge> = face
        .edges
        .iter()
        .map(|e| b::Edge {
            curve: e.curve.clone(),
            nurbs: e.nurbs.cloned(),
            interval: e.interval,
            vertices: e.vertices,
            tolerance: e.tolerance,
            uses: Vec::new(),
        })
        .collect();
    let mut bare = HashSet::new();
    let mut loops = Vec::with_capacity(face.loops.len());
    for (li, l) in face.loops.iter().enumerate() {
        let li = u32::try_from(li).map_err(|_| "too many loops")?;
        let mut uses = Vec::with_capacity(l.len());
        for (p, u) in l.iter().enumerate() {
            let p = u32::try_from(p).map_err(|_| "too many uses")?;
            let edge = edges.get_mut(u.edge as usize).ok_or("edge reference")?;
            edge.uses.push(EdgeUseRef { face: 0, r#loop: li, position: p, upper: !u.forward });
            let pcurve = match &u.pcurve {
                Some(pc) => pc.clone(),
                None => {
                    bare.insert((0, li, p));
                    // Not read: the use goes without a trim (`arrays_with_bare_uses`).
                    Curve2::Spline { knots: vec![edge.interval[0], edge.interval[0], edge.interval[1], edge.interval[1]], points: vec![[0.0; 2]; 2] }
                }
            };
            uses.push(OrientedEdge { edge: u.edge, forward: u.forward, pcurve });
        }
        loops.push(Loop::Edges { edges: uses });
    }
    let vertices = face.vertices.iter().map(|&point| Vertex { point, tolerance: face.tolerance }).collect();
    let record = b::Brep {
        version: b::VERSION,
        tolerance: face.tolerance,
        regions: vec![Region { shells: vec![0], void: false }],
        shells: vec![Shell::Faces { faces: vec![OrientedFace { face: 0, forward: true }], wires: Vec::new() }],
        faces: vec![b::Face { surface: face.surface.clone(), nurbs: face.nurbs.cloned(), domain: face.domain, loops }],
        edges,
        vertices,
    };
    Ok((record, bare))
}

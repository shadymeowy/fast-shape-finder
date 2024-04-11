import numpy as np
import cython
from typing import NamedTuple
from cython.parallel import prange
from cython cimport view
from libc.math cimport pow, sqrt, atan, pi, sin, cos
from libc.time cimport time
from libc.stdlib cimport srand, rand
from libc.stdint cimport uint32_t, uint8_t

ctypedef unsigned int uint
ctypedef unsigned char uchar

DEF LCG_A = 1664525
DEF LCG_C = 1013904223
srand(time(NULL))


class LCHBounds(NamedTuple):
    M: np.ndarray
    i: tuple
    c: tuple


cdef:
    struct lch_bounds:
        int[2] i
        int[2] c
        int[9] M

    struct Circle:
        int x
        int y
        int r

    struct Ellipse:
        double x
        double y
        double a
        double b
        double theta

    struct Line:
        int px
        int py
        int nx
        int ny
    
    struct Line2:
        Line l1
        Line l2


def calculate_lch_bounds(hue_limit, chroma_limit, intensity_limit, rgb_bits=8):
    theta1 = hue_limit[0]
    theta2 = hue_limit[1]
    c1 = chroma_limit[0] / np.cos((theta2 - theta1) / 2)
    c2 = chroma_limit[1] / np.cos((theta2 - theta1) / 2)
    i1 = intensity_limit[0]
    i2 = intensity_limit[1]

    T1 = np.array([[1, -1 / 2, -1 / 2],
                   [0, np.sqrt(3)/2, -np.sqrt(3) / 2],
                   [1 / 3, 1 / 3, 1 / 3]])
    T2 = np.array([[np.sin(theta2), -np.cos(theta2), 0],
                   [-np.sin(theta1), np.cos(theta1), 0],
                   [0, 0, np.sin(theta2 - theta1)]])
    T2 /= np.sin(theta2 - theta1)
    M = T2 @ T1

    if rgb_bits is not None:
        M *= 1 << rgb_bits
        M = np.round(M).astype(np.int32)
        rgb_bits2 = rgb_bits << 1
        c1 *= 1 << rgb_bits2
        c1 = int(round(c1))
        c2 *= 1 << rgb_bits2
        c2 = int(round(c2))
        i1 *= 1 << rgb_bits2
        i1 = int(round(i1))
        i2 *= 1 << rgb_bits2
        i2 = int(round(i2))

    return LCHBounds(M.flatten(), (i1, i2), (c1, c2))


def mask_image_py(image, bounds):
    M, i, c = bounds
    M = M.reshape(3, 3)
    image_shape = image.shape
    image = image.reshape(-1, 3).astype(np.int32)
    image = (M @ image.T).T
    mask = (image[:, 0] > 0) & (image[:, 1] > 0)
    mask &= (image[:, 2] > i[0]) & (image[:, 2] < i[1])
    mask &= (image[:, 0] + image[:, 1] < c[1])
    mask &= (image[:, 0] + image[:, 1] > c[0])
    mask = mask.reshape(image_shape[:-1])
    result = mask.astype(np.uint8) * 255
    return result


def mask_image(image, bounds, buffer=None):
    if buffer is None:
        buffer = np.zeros(image.shape[:-1], dtype=np.uint8)
    if isinstance(bounds, LCHBounds):
        bounds = bounds._asdict()
    mask_image_cy_impl(image, buffer, bounds)
    return buffer


def selective_grayscale(image, bounds, buffer=None):
    if buffer is None:
        buffer = np.zeros(image.shape[:-1], dtype=np.uint8)
    if isinstance(bounds, LCHBounds):
        bounds = bounds._asdict()
    selective_grayscale_impl(image, buffer, bounds)
    return buffer


def mask_color(color, bounds):
    M, i, c = bounds
    M = M.reshape(3, 3)
    color = np.array(color).astype(np.int32)
    color = (M @ color.T).T
    mask = (color[0] > 0) & (color[1] > 0)
    mask &= (color[2] > i[0]) & (color[2] < i[1])
    mask &= (color[0] + color[1] < c[1])
    mask &= (color[0] + color[1] > c[0])
    return mask


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline int _color_mask(unsigned char c[3], lch_bounds *b) noexcept nogil:
    cdef int v[3]
    v[0] = b.M[0]*c[0] + b.M[1]*c[1] + b.M[2]*c[2]
    v[1] = b.M[3]*c[0] + b.M[4]*c[1] + b.M[5]*c[2]
    v[2] = b.M[6]*c[0] + b.M[7]*c[1] + b.M[8]*c[2]
    cdef int tmp = v[0] + v[1]
    cdef int result = ((v[0] > 0) and (v[1] > 0)
                    and (v[2] > b.i[0]) and (v[2] < b.i[1])
                    and (tmp > b.c[0]) and (tmp < b.c[1]))
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline int _color_selective(unsigned char c[3], lch_bounds *b) noexcept nogil:
    cdef int v[3]
    v[0] = b.M[0]*c[0] + b.M[1]*c[1] + b.M[2]*c[2]
    v[1] = b.M[3]*c[0] + b.M[4]*c[1] + b.M[5]*c[2]
    v[2] = b.M[6]*c[0] + b.M[7]*c[1] + b.M[8]*c[2]
    cdef int tmp = v[0] + v[1]
    cdef int result = ((v[0] > 0) and (v[1] > 0)
                    and (v[2] > b.i[0]) and (v[2] < b.i[1])
                    and (tmp > b.c[0]) and (tmp < b.c[1]))
    return (v[2] >> 8) if result else 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.profile(True)
cpdef unsigned int mask_image_cy_impl(
        unsigned char[:,:,::view.contiguous] _image,
        unsigned char[:,::view.contiguous] _output,
        lch_bounds bounds) nogil:
    cdef:
        unsigned n = _image.shape[0]*_image.shape[1]
        unsigned char* _image2 = &_image[0, 0, 0]
        unsigned char[3]* image = <unsigned char[3]*>_image2
        unsigned char* output = &_output[0, 0]
        unsigned i
        lch_bounds *b = &bounds

    for i in prange(n):
        output[i] = _color_mask(image[i], b)

    return n


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.profile(True)
cpdef unsigned int selective_grayscale_impl(
        unsigned char[:,:,::view.contiguous] _image,
        unsigned char[:,::view.contiguous] _output,
        lch_bounds bounds) nogil:
    cdef:
        unsigned n = _image.shape[0]*_image.shape[1]
        unsigned char* _image2 = &_image[0, 0, 0]
        unsigned char[3]* image = <unsigned char[3]*>_image2
        unsigned char* output = &_output[0, 0]
        unsigned i
        lch_bounds *b = &bounds

    for i in prange(n):
        output[i] = _color_selective(image[i], b)

    return n


cdef inline uint32_t reverse_bits(uint32_t n) noexcept nogil:
    # n = (n << 16) | (n >> 16)
    n = ((n & 0x00ff00ffu) << 8) | ((n & 0xff00ff00u) >> 8)
    n = ((n & 0x0f0f0f0fu) << 4) | ((n & 0xf0f0f0f0u) >> 4)
    n = ((n & 0x33333333u) << 2) | ((n & 0xccccccccu) >> 2)
    n = ((n & 0x55555555u) << 1) | ((n & 0xaaaaaaaau) >> 1)
    return n


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(True)
cpdef int simple_edge_hsv(uchar[:,:,::view.contiguous] _image, int[:,::view.contiguous] _ps, lch_bounds bounds, int row_count, int column_count):
    cdef:
        int x, y, o, t, i
        int h = _image.shape[0]
        int w = _image.shape[1]
        int buff_size = _ps.shape[0]
        uchar[3]* image = <uchar[3]*>&_image[0, 0, 0]
        int[2]* ps = <int[2]*>&_ps[0, 0]
        int j = 0

    for i in range(row_count):
        y = (reverse_bits(i)*h)>>16
        o = y*w
        flag = 0
        for x in range(w):
            t = _color_mask(image[o+x], &bounds)
            if t and not flag:
                if j >= buff_size:
                    return j
                ps[j] = [x, y]
                j += 1
                flag = 1
            if not t and flag:
                if j >= buff_size:
                    return j
                ps[j] = [x, y]
                j += 1
                flag = 0
        if flag:
            ps[j] = [x, y]
            j += 1
        
    for i in range(column_count):
        x = (reverse_bits(i)*w)>>16
        flag = 0
        for y in range(h):
            t = _color_mask(image[y*w+x], &bounds)
            if t and not flag:
                if j >= buff_size:
                    return j
                ps[j] = [x, y]
                j += 1
                flag = 1
            if not t and flag:
                if j >= buff_size:
                    return j
                ps[j] = [x, y]
                j += 1
                flag = 0
        if flag:
            ps[j] = [x, y]
            j += 1
    return j


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(True)
cpdef int simple_edge(unsigned char[:,::view.contiguous] _image, int[:,::view.contiguous] _ps, int row_count, int column_count):
    cdef:
        int x, y, o, t, i
        int h = _image.shape[0]
        int w = _image.shape[1]
        int buff_size = _ps.shape[0]
        uchar *image = &_image[0, 0]
        int[2]* ps = <int[2]*>&_ps[0, 0]
        int j = 0

    for i in range(row_count):
        y = (reverse_bits(i)*h)>>16
        o = y*w
        flag = 0
        for x in range(w):
            t = image[o+x] > 0
            if t and not flag:
                if j >= buff_size:
                    return j
                ps[j] = [x, y]
                j += 1
                flag = 1
            if not t and flag:
                if j >= buff_size:
                    return j
                ps[j] = [x, y]
                j += 1
                flag = 0
        if flag:
            ps[j] = [x, y]
            j += 1
        
    for i in range(column_count):
        x = (reverse_bits(i)*w)>>16
        flag = 0
        for y in range(h):
            t = image[y*w+x]
            if t and not flag:
                if j >= buff_size:
                    return j
                ps[j] = [x, y]
                j += 1
                flag = 1
            if not t and flag:
                if j >= buff_size:
                    return j
                ps[j] = [x, y]
                j += 1
                flag = 0
        if flag:
            ps[j] = [x, y]
            j += 1
    return j


@cython.wraparound(False)
@cython.nonecheck(False)
@cython.profile(True)
cpdef int simple_denoise(int[:,::view.contiguous] _ps_in, int[:,::view.contiguous] _ps_out, int min_distance):
    cdef:
        int in_size = _ps_in.shape[0]
        int out_size = _ps_in.shape[0]
    if in_size == 0:
        return 0
    cdef:
        int[2]* ps_in = <int[2]*>&_ps_in[0, 0]
        int[2]* ps_out = <int[2]*>&_ps_out[0, 0]
        int head[2]
        int tail[2]
        bint flag
        int i, j
    j = 0
    head = ps_in[0]
    tail = ps_in[1]
    for i in range(2, in_size - 1, 2):
        if ps_in[i][1] == tail[1] and ps_in[i][0] - tail[0] < min_distance:
            tail = ps_in[i+1]
        elif ps_in[i][0] == tail[0] and ps_in[i][1] - tail[1] < min_distance:
            tail = ps_in[i+1]
        else:
            if (tail[1] == head[1] and tail[0] - head[0] >= min_distance) or (tail[0] == head[0] and tail[1] - head[1] >= min_distance):
                ps_out[j] = head
                ps_out[j+1] = tail
                j += 2
            head = ps_in[i]
            tail = ps_in[i+1]
    return j


@cython.wraparound(False)
@cython.nonecheck(False)
@cython.profile(True)
cpdef int remove_inliers(int[:,::view.contiguous] _ps_in, int[:,::view.contiguous] _ps_out, uint8_t[::view.contiguous] _ms):
    cdef:
        int in_size = _ps_in.shape[0]
        int out_size = _ps_in.shape[0]
        uint8_t* ms = &_ms[0]
    if in_size == 0:
        return 0
    cdef:
        int[2]* ps_in = <int[2]*>&_ps_in[0, 0]
        int[2]* ps_out = <int[2]*>&_ps_out[0, 0]
        int i, j
    j = 0
    for i in range(in_size):
        if not ms[i]:
            ps_out[j] = ps_in[i]
            j += 1
    return j


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef int solve_matrix(double m[5][6]) noexcept nogil:
    cdef:
        int i, e, j, k
        double s
    for i in range(5):
        for e in range(i + 1, 5):
            if m[i][i] == 0 or (m[i][i] < m[e][i] and m[e][i] != 0):
                for k in range(6):
                    s = m[i][k]
                    m[i][k] = m[e][k]
                    m[e][k] = s
        t = m[i][i]
        if t == 0:
            return 1
        for e in range(i, 6):
            m[i][e] /= t
        for e in range(i + 1, 5):
            t = m[e][i]
            for j in range(i, 6):
                m[e][j] -= m[i][j] * t
    for i in reversed(range(1, 5)):
        for e in range(i):
            for j in range(5, 6):
                m[e][j] -= m[i][j] * m[e][i]
            m[e][i] = 0
    return 0


@cython.cdivision(True)
cdef inline int points_to_circle(int p[3][2], Circle *r) noexcept nogil: # Benchmark will it be faster to use solve_matrix
    cdef:
        int c1 = (p[0][0]*p[0][0]+p[0][1]*p[0][1])
        int c2 = (p[1][0]*p[1][0]+p[1][1]*p[1][1])
        int c3 = (p[2][0]*p[2][0]+p[2][1]*p[2][1])
        int A=p[0][0]*(p[1][1]-p[2][1])-p[0][1]*(p[1][0]-p[2][0])+p[1][0]*p[2][1]-p[2][0]*p[1][1]
        int A2=2*A
        int B=c1*(p[2][1]-p[1][1])+c2*(p[0][1]-p[2][1])+c3*(p[1][1]-p[0][1])
        int C=c1*(p[1][0]-p[2][0])+c2*(p[2][0]-p[0][0])+c3*(p[0][0]-p[1][0])
        int D=c1*(p[2][0]*p[1][1]-p[1][0]*p[2][1])+c2*(p[0][0]*p[2][1]-p[2][0]*p[0][1])+c3*(p[1][0]*p[0][1]-p[0][0]*p[1][1])
    if not A:
        return 1
    r.x=-B//A2
    r.y=-C//A2
    r.r=<int>sqrt(<double>(r.x*r.x+r.y*r.y-D/A)) #TODO
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(True)
cpdef Circle ransac_circle(int[:,::view.contiguous] _ps, int count, int min_r, int max_r, int delta) nogil:
    cdef:
        int[2]* ps = <int[2]*>&_ps[0, 0]
        uint i = rand()
        int n = _ps.shape[0]
        int j, k, t, l
        Circle best = Circle(0, 0, 0)
        Circle curr
        int best_score = 0
        int curr_score
        int p[3][2]
        int pi[3]
        
    for l in range(count):
        j = 0
        while j < 3:
            t = i%n
            for k in range(j):
                if pi[k] == t:
                    break
            else:
                pi[j] = t
                p[j][0] = ps[t][0]
                p[j][1] = ps[t][1]
                j += 1
            i *= LCG_A
            i += LCG_C
        if points_to_circle(p, &curr):
            continue
        if curr.r > max_r or curr.r < min_r:
            continue
        curr_score = 0
        for k in range(n):
            if (curr.r-delta)*(curr.r-delta)<=((ps[k][0]-curr.x)*(ps[k][0]-curr.x)+(ps[k][1]-curr.y)*(ps[k][1]-curr.y))<=(curr.r+delta)*(curr.r+delta):
                curr_score += 1
        if curr_score > best_score:
            best_score = curr_score
            best = curr
        
    return best


@cython.cdivision(True)
cdef inline int points_to_ellipse(int p[5][2], Ellipse *r) noexcept nogil:
    cdef:
        double[5][6] m
        double b[5]
        double dnt, det, t1
        int i
    for i in range(5):
        m[i][0] = p[i][0]*p[i][0]
        m[i][1] = p[i][0]*p[i][1]
        m[i][2] = p[i][1]*p[i][1]
        m[i][3] = p[i][0]
        m[i][4] = p[i][1]
        m[i][5] = 1
    if solve_matrix(m):
        return 1
    for i in range(5):
        b[i] = m[i][5]
    dnt = b[1]*b[1] - 4*b[0]*b[2]
    if dnt >= 0:
        return 2
    det = 2*(b[0]*b[4]*b[4]+b[2]*b[3]*b[3]-b[1]*b[3]*b[4]-dnt)
    if dnt > 0:
        return 3
    t1 = sqrt((b[0]-b[2])*(b[0]-b[2])+b[1]*b[1])
    r.a = -sqrt(det*(b[0]+b[2]+t1))/dnt
    r.b = -sqrt(det*(b[0]+b[2]-t1))/dnt
    r.x = (2*b[2]*b[3]-b[1]*b[4])/dnt
    r.y = (2*b[0]*b[4]-b[1]*b[3])/dnt
    if b[1] == 0:
        r.theta = 0 if b[0] < b[2] else pi/2
    else:
        r.theta = atan((b[2]-b[0]-t1)/b[1])
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void ellipse_coeffs(Ellipse e, double[6] c) noexcept nogil:
    cdef:
        double sn = sin(e.theta)
        double cs = cos(e.theta)
    c[0] = (e.a*sn)*(e.a*sn)+(e.b*cs)*(e.b*cs)
    c[1] = 2*(e.b*e.b-e.a*e.a)*sn*cs
    c[2] = (e.a*cs)*(e.a*cs)+(e.b*sn)*(e.b*sn)
    c[3] = -2*c[0]*e.x-c[1]*e.y
    c[4] = -c[1]*e.x-2*c[2]*e.y
    c[5] = c[0]*e.x*e.x+c[1]*e.x*e.y+c[2]*e.y*e.y-e.a*e.a*e.b*e.b


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double ellipse_eval(int p[2], double[6] c) noexcept nogil:
    return c[0]*p[0]*p[0]+c[1]*p[0]*p[1]+c[2]*p[1]*p[1]+c[3]*p[0]+c[4]*p[1]+c[5]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(True)
cpdef Ellipse ransac_ellipse(int[:,::view.contiguous] _ps, int count, int min_r, int max_r, double ratio, int delta):
    cdef:
        int[2]* ps = <int[2]*>&_ps[0, 0]
        uint i = rand()
        int n = _ps.shape[0]
        int j, k, t, l
        Ellipse best = Ellipse(0, 0, 0, 0, 0)
        Ellipse curr
        int best_score = 0
        int curr_score
        int p[5][2]
        int pi[5]
        double b1[6], b2[6]
        
    for l in range(count):
        j = 0
        while j < 5:
            t = i%n
            for k in range(j):
                if pi[k] == t:
                    break
            else:
                pi[j] = t
                p[j][0] = ps[t][0]
                p[j][1] = ps[t][1]
                j += 1
            i *= LCG_A
            i += LCG_C
        if points_to_ellipse(p, &curr):
            continue
        if curr.b > curr.a*ratio or curr.a > max_r or curr.a < min_r or curr.b > max_r or curr.b < min_r:
            continue
        curr.a -= delta
        curr.b -= delta
        ellipse_coeffs(curr, b1)
        curr.a += 2*delta
        curr.b += 2*delta
        ellipse_coeffs(curr, b2)
        curr.a -= delta
        curr.b -= delta
        curr_score = 0
        for k in range(n):
            if ellipse_eval(ps[k], b1) >= 0 and ellipse_eval(ps[k], b2) <= 0:
                curr_score += 1
        if curr_score > best_score:
            best_score = curr_score
            best = curr
        
    return best


@cython.cdivision(True)
cdef inline int points_to_line(int p[2][2], Line *l) noexcept nogil:
    if p[0][0] == p[1][0] and p[0][1] == p[1][1]:
        return 1
    l.px = p[0][0] 
    l.py = p[0][1]
    l.nx = p[1][1] - p[0][1]
    l.ny = p[0][0] - p[1][0]
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void line_coeffs(Line l, int[3] c) noexcept nogil:
    c[0] = l.nx
    c[1] = l.ny
    c[2] = -l.px*l.nx - l.py*l.ny


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline int line_eval(int p[2], int[3] c) noexcept nogil:
    return c[0]*p[0]+c[1]*p[1]+c[2]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline void line_shift(Line l, int[3] c, int delta) noexcept nogil:
    c[2] += <int>(delta*sqrt(l.nx*l.nx+l.ny*l.ny))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(True)
cpdef Line ransac_line(int[:,::view.contiguous] _ps, uint8_t[::view.contiguous] _ms, int count, int delta, double slope_limit):
    cdef:
        int[2]* ps = <int[2]*>&_ps[0, 0]
        uint8_t* ms = &_ms[0]
        uint i = rand()
        int n = _ps.shape[0]
        int j, k, t, l
        Line best = Line(0, 0, 0, 0)
        Line curr
        int best_score = 0
        int curr_score
        int p[2][2]
        int pi[2]
        int b1[3], b2[3]
        
    for l in range(count):
        j = 0
        while j < 2:
            t = i%n
            for k in range(j):
                if pi[k] == t:
                    break
            else:
                pi[j] = t
                p[j][0] = ps[t][0]
                p[j][1] = ps[t][1]
                j += 1
            i *= LCG_A
            i += LCG_C
        if points_to_line(p, &curr):
            continue
        if (curr.nx*curr.nx) <= slope_limit*slope_limit*(curr.nx*curr.nx+curr.ny*curr.ny):
            continue
        line_coeffs(curr, b1)
        line_coeffs(curr, b2)
        line_shift(curr, b1, delta)
        line_shift(curr, b2, -delta)
        curr_score = 0
        for k in range(n):
            if line_eval(ps[k], b1) >= 0 and line_eval(ps[k], b2) <= 0:
                curr_score += 1
        if curr_score > best_score:
            best_score = curr_score
            best = curr
    
    line_coeffs(best, b1)
    line_coeffs(best, b2)
    line_shift(best, b1, delta)
    line_shift(best, b2, -delta)
    for k in range(n):
        if line_eval(ps[k], b1) >= 0 and line_eval(ps[k], b2) <= 0:
            ms[k] = 1
        else:
            ms[k] = 0
    return best


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(True)
cpdef tuple ransac_line2(int[:,::view.contiguous] _ps, uint8_t[::view.contiguous] _ms, int count, double ratio, int delta):
    cdef:
        int[2]* ps = <int[2]*>&_ps[0, 0]
        uint8_t* ms = &_ms[0]
        uint i = rand()
        int n = _ps.shape[0]
        int j, k, t, l
        Line2 best = Line2(Line(0, 0, 0, 0), Line(0, 0, 0, 0))
        Line2 curr
        int best_score = 0
        int curr_score
        int p[4][2]
        int pi[4]
        int b11[3], b12[3], b21[3], b22[3]
        int ncos, n1_l, n2_l
        int f1, f2
    
    for l in range(count):
        j = 0
        while j < 4:
            t = i%n
            for k in range(j):
                if pi[k] == t:
                    break
            else:
                pi[j] = t
                p[j][0] = ps[t][0]
                p[j][1] = ps[t][1]
                j += 1
            i *= LCG_A
            i += LCG_C
        if points_to_line(p, &curr.l1):
            continue
        if points_to_line((p + 2), &curr.l2):
            continue
        ncos = curr.l1.nx*curr.l2.nx + curr.l1.ny*curr.l2.ny
        n1_l = curr.l1.nx*curr.l1.nx + curr.l1.ny*curr.l1.ny
        n2_l = curr.l2.nx*curr.l2.nx + curr.l2.ny*curr.l2.ny
        if ncos*ncos < ratio*ratio*n1_l*n2_l:
            continue
        line_coeffs(curr.l1, b11)
        line_coeffs(curr.l1, b12)
        line_shift(curr.l1, b11, delta)
        line_shift(curr.l1, b12, -delta)
        line_coeffs(curr.l2, b21)
        line_coeffs(curr.l2, b22)
        line_shift(curr.l2, b21, delta)
        line_shift(curr.l2, b22, -delta)
        curr_score = 0
        for k in range(n):
            f1 = (line_eval(ps[k], b11) >= 0 and line_eval(ps[k], b12) <= 0)
            f2 = (line_eval(ps[k], b21) >= 0 and line_eval(ps[k], b22) <= 0)
            if f1 ^ f2:
                curr_score += 1
        if curr_score > best_score:
            best_score = curr_score
            best = curr

    line_coeffs(best.l1, b11)
    line_coeffs(best.l1, b12)
    line_shift(best.l1, b11, delta)
    line_shift(best.l1, b12, -delta)
    line_coeffs(best.l2, b21)
    line_coeffs(best.l2, b22)
    line_shift(best.l2, b21, delta)
    line_shift(best.l2, b22, -delta)
    for k in range(n):
            f1 = (line_eval(ps[k], b11) >= 0 and line_eval(ps[k], b12) <= 0)
            f2 = (line_eval(ps[k], b21) >= 0 and line_eval(ps[k], b22) <= 0)
            if f1 or f2:
                ms[k] = 1
            else:
                ms[k] = 0
    return best.l1, best.l2
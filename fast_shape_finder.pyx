import cython
from cython cimport view
from libc.math cimport pow, sqrt, atan, pi, sin, cos
from libc.time cimport time
from libc.stdlib cimport srand, rand

ctypedef unsigned int uint
ctypedef unsigned char uchar

DEF LCG_A = 1664525
DEF LCG_C = 1013904223
srand(time(NULL))

cdef:
    struct hsv_bounds:
        unsigned primary_index
        unsigned secondary_index_1
        unsigned secondary_index_2
        unsigned value
        unsigned saturation_numerator
        unsigned saturation_denominator
        unsigned hue_1_numerator
        unsigned hue_1_denominator
        unsigned hue_2_numerator
        unsigned hue_2_denominator
        
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef bint is_primary_color(unsigned char _c[3], hsv_bounds *b) nogil:
    cdef:
        unsigned c[3]
    if (
        _c[b.primary_index] < b.value
        or _c[b.primary_index] < _c[b.secondary_index_1]
        or _c[b.primary_index] < _c[b.secondary_index_2]
    ):
        return False
    c[0] = _c[b.primary_index]
    if _c[b.secondary_index_1] > _c[b.secondary_index_2]:
        c[1] = _c[b.secondary_index_1]
        c[2] = _c[b.secondary_index_2]
        return (
            c[2]*b.saturation_denominator <= c[0]*b.saturation_numerator
            and (c[1] - c[2])*b.hue_1_denominator <= (c[0] - c[2])*b.hue_1_numerator
        )
    else:
        c[1] = _c[b.secondary_index_2]
        c[2] = _c[b.secondary_index_1]
        return (
            c[2]*b.saturation_denominator <= c[0]*b.saturation_numerator
            and (c[1] - c[2])*b.hue_2_denominator <= (c[0] - c[2])*b.hue_2_numerator
        )

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.profile(True)
cpdef uint hsv_mask(uchar[:,:,::view.contiguous] _image, uchar[:,::view.contiguous] _output, hsv_bounds bounds):
    cdef:
        uint n = _image.shape[0]*_image.shape[1]
        unsigned char* _image2 = &_image[0, 0, 0]
        unsigned char[3]* image = <unsigned char[3]*>_image2
        unsigned char* output = &_output[0, 0]
        
    for i in range(n):
        output[i] = 255*is_primary_color(image[i], &bounds)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(True)
cpdef uint simple_edge_hsv(uchar[:,:,::view.contiguous] _image, int[:,::view.contiguous] _ps, hsv_bounds bounds, uint row_count, uint column_count):
    cdef:
        uint i = rand()
        uint x, y, o, t
        uint h = _image.shape[0]
        uint w = _image.shape[1]
        uint buff_size = _ps.shape[0]
        uchar[3]* image = <uchar[3]*>&_image[0, 0, 0]
        int[2]* ps = <int[2]*>&_ps[0, 0]
        uint j = 0

    for _ in range(row_count):
        y = i%h
        o = y*w
        flag = 0
        for x in range(w):
            t = is_primary_color(image[o+x], &bounds)
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
        i *= LCG_A
        i += LCG_C
        
    for _ in range(column_count):
        x = i%w
        flag = 0
        for y in range(h):
            t = is_primary_color(image[y*w+x], &bounds)
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
        i *= LCG_A
        i += LCG_C
    return j

cdef struct Circle:
    int x
    int y
    int r

@cython.wraparound(False)
@cython.nonecheck(False)
@cython.profile(True)
cpdef uint simple_denoise(int[:,::view.contiguous] _ps_in, int[:,::view.contiguous] _ps_out, uint min_distance):
    cdef:
        uint in_size = _ps_in.shape[0]
        uint out_size = _ps_in.shape[0]
    if in_size == 0:
        return 0
    cdef:
        int[2]* ps_in = <int[2]*>&_ps_in[0, 0]
        int[2]* ps_out = <int[2]*>&_ps_out[0, 0]
        int head[2]
        int tail[2]
        bint flag
        uint i, j
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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(True)
cpdef Circle ransac_circle(int[:,::view.contiguous] _ps, uint count, int min_r, int max_r, uint delta) nogil:
    cdef:
        int[2]* ps = <int[2]*>&_ps[0, 0]
        uint i = rand()
        uint n = _ps.shape[0]
        uint j, k, t, l
        Circle best = Circle(0, 0, 0)
        Circle curr
        uint best_score = 0
        uint curr_score
        uint p[3][2]
        uint pi[3]
        
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
cdef inline uint points_to_circle(uint p[3][2], Circle *r) nogil: # Benchmark will it be faster to use solve_matrix
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
    r.r=<uint>sqrt(<double>(r.x*r.x+r.y*r.y-D/A)) #TODO
    return 0

cdef struct Ellipse:
    double x
    double y
    double a
    double b
    double theta
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef int solve_matrix(double m[5][6]) nogil:
    cdef:
        int i, e, j, k
        double s
    for i in range(5):
        if m[i][i]==0:
            for e in range(i + 1, 5):
                if m[e][i]!=0:
                    for k in range(6):
                        s = m[i][k]
                        m[i][k] = m[e][k]
                        m[e][k] = s
                    break
            else:
                return 1
        t = m[i][i]
        for e in range(i,6):
            m[i][e] /= t
        for e in range(i + 1,5):
            t = m[e][i]
            for j in range(i,6):
                m[e][j] -= m[i][j] * t
    for i in reversed(range(1,5)):
        for e in range(i):
            for j in range(5,6):
                m[e][j] -= m[i][j] * m[e][i]
            m[e][i] = 0
    return 0

@cython.cdivision(True)
cdef inline uint points_to_ellipse(uint p[5][2], Ellipse *r) nogil:
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
cdef inline void ellipse_coeffs(Ellipse e, double[6] c) nogil:
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
cdef inline double ellipse_eval(double x, double y, double[6] c) nogil:
    return c[0]*x*x+c[1]*x*y+c[2]*y*y+c[3]*x+c[4]*y+c[5]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.profile(True)
cpdef Ellipse ransac_ellipse(int[:,::view.contiguous] _ps, uint count, int min_r, int max_r, double ratio, uint delta):
    cdef:
        int[2]* ps = <int[2]*>&_ps[0, 0]
        uint i = rand()
        uint n = _ps.shape[0]
        uint j, k, t, l
        Ellipse best = Ellipse(0, 0, 0, 0, 0)
        Ellipse curr
        uint best_score = 0
        uint curr_score
        uint p[5][2]
        uint pi[5]
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
            if ellipse_eval(ps[k][0], ps[k][1], b1) >= 0 and ellipse_eval(ps[k][0], ps[k][1], b2) <= 0:
                curr_score += 1
        if curr_score > best_score:
            best_score = curr_score
            best = curr
        
    return best
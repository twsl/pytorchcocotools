#include <stdlib.h>
#include <math.h>
#include <stdio.h>

typedef unsigned int uint;
typedef unsigned long siz;
// typedef unsigned chat byte;
typedef double *BB;
typedef struct
{
    siz h, w, m;
    uint *cnts;
} RLE;

void rleInit(RLE *R, siz h, siz w, siz m, uint *cnts)
{
    R->h = h;
    R->w = w;
    R->m = m;
    R->cnts = (m == 0) ? 0 : malloc(sizeof(uint) * m);
    siz j;
    if (cnts)
        for (j = 0; j < m; j++)
            R->cnts[j] = cnts[j];
}

uint umin(uint a, uint b) { return (a < b) ? a : b; }
uint umax(uint a, uint b) { return (a > b) ? a : b; }

void rleFree(RLE *R)
{
    free(R->cnts);
    R->cnts = 0;
}
int uintCompare(const void *a, const void *b)
{
    uint c = *((uint *)a), d = *((uint *)b);
    return c > d ? 1 : c < d ? -1
                             : 0;
}

void rleFrPoly(RLE *R, const double *xy, siz k, siz h, siz w)
{
    /* upsample and get discrete points densely along entire boundary */
    siz j, m = 0;
    double scale = 5;
    int *x, *y, *u, *v;
    uint *a, *b;
    x = malloc(sizeof(int) * (k + 1));
    y = malloc(sizeof(int) * (k + 1));
    for (j = 0; j < k; j++)
        x[j] = (int)(scale * xy[j * 2 + 0] + .5);
    x[k] = x[0];
    for (j = 0; j < k; j++)
        y[j] = (int)(scale * xy[j * 2 + 1] + .5);
    y[k] = y[0];
    for (j = 0; j < k; j++)
        m += umax(abs(x[j] - x[j + 1]), abs(y[j] - y[j + 1])) + 1;
    u = malloc(sizeof(int) * m);
    v = malloc(sizeof(int) * m);
    m = 0;
    for (j = 0; j < k; j++)
    {
        int xs = x[j], xe = x[j + 1], ys = y[j], ye = y[j + 1], dx, dy, t, d;
        int flip;
        double s;
        dx = abs(xe - xs);
        dy = abs(ys - ye);
        flip = (dx >= dy && xs > xe) || (dx < dy && ys > ye);
        if (flip)
        {
            t = xs;
            xs = xe;
            xe = t;
            t = ys;
            ys = ye;
            ye = t;
        }
        s = dx >= dy ? (double)(ye - ys) / dx : (double)(xe - xs) / dy;
        if (dx >= dy)
            for (d = 0; d <= dx; d++)
            {
                t = flip ? dx - d : d;
                u[m] = t + xs;
                v[m] = (int)(ys + s * t + .5);
                m++;
            }
        else
            for (d = 0; d <= dy; d++)
            {
                t = flip ? dy - d : d;
                v[m] = t + ys;
                u[m] = (int)(xs + s * t + .5);
                m++;
            }
    }
    /* get points along y-boundary and downsample */
    free(x);
    free(y);
    k = m;
    m = 0;
    double xd, yd;
    x = malloc(sizeof(int) * k);
    y = malloc(sizeof(int) * k);
    for (j = 1; j < k; j++)
        if (u[j] != u[j - 1])
        {
            xd = (double)(u[j] < u[j - 1] ? u[j] : u[j] - 1);
            xd = (xd + .5) / scale - .5;
            if (floor(xd) != xd || xd < 0 || xd > w - 1)
                continue;
            yd = (double)(v[j] < v[j - 1] ? v[j] : v[j - 1]);
            yd = (yd + .5) / scale - .5;
            if (yd < 0)
                yd = 0;
            else if (yd > h)
                yd = h;
            yd = ceil(yd);
            x[m] = (int)xd;
            y[m] = (int)yd;
            m++;
        }
    /* compute rle encoding given y-boundary points */
    k = m;
    a = malloc(sizeof(uint) * (k + 1));
    for (j = 0; j < k; j++)
        a[j] = (uint)(x[j] * (int)(h) + y[j]);
    a[k++] = (uint)(h * w);
    free(u);
    free(v);
    free(x);
    free(y);
    qsort(a, k, sizeof(uint), uintCompare);
    uint p = 0;
    for (j = 0; j < k; j++)
    {
        uint t = a[j];
        a[j] -= p;
        p = t;
    }
    b = malloc(sizeof(uint) * k);
    j = m = 0;
    b[m++] = a[j++];
    while (j < k)
        if (a[j] > 0)
            b[m++] = a[j++];
        else
        {
            j++;
            if (j < k)
                b[m - 1] += a[j++];
        }
    rleInit(R, h, w, m, b);
    free(a);
    free(b);
}

void rleFrString(RLE *R, char *s, siz h, siz w)
{
    siz m = 0, p = 0, k;
    long x;
    int more;
    uint *cnts;
    while (s[m])
        m++;
    cnts = malloc(sizeof(uint) * m);
    m = 0;
    while (s[p])
    {
        x = 0;
        k = 0;
        more = 1;
        while (more)
        {
            char c = s[p] - 48;
            x |= (c & 0x1f) << 5 * k;
            more = c & 0x20;
            p++;
            k++;
            if (!more && (c & 0x10))
                x |= -1 << 5 * k;
        }
        if (m > 2)
            x += (long)cnts[m - 2];
        cnts[m++] = (uint)x;
    }
    rleInit(R, h, w, m, cnts);
    free(cnts);
}

int main()
{
    // Create an instance of RLE
    RLE *rle_instance = (RLE *)malloc(sizeof(RLE));

    // Initialize h, w, and m if needed, here set to 0 as an example
    rle_instance->h = 427;
    rle_instance->w = 640;
    rle_instance->m = 2;

    // Define the cnts data
    double cnts_data[] = {
        266.83,
        189.37,
        267.79,
        175.29,
        269.46,
        170.04,
        271.37,
        165.98,
        270.89,
        163.12,
        269.12,
        159.54,
        272.8,
        156.44,
        287.36,
        156.44,
        293.33,
        157.87,
        296.91,
        160.49,
        296.91,
        161.21,
        291.89,
        161.92,
        289.98,
        165.03,
        291.42,
        169.56,
        285.16,
        196.54,
    };
    size_t cnts_size = sizeof(cnts_data) / sizeof(cnts_data[0]);
    double *cnts = (double *)malloc(cnts_size * sizeof(double));
    for (size_t i = 0; i < cnts_size; i++)
    {
        cnts[i] = cnts_data[i];
    }

    // ... use the rle_instance as needed ...
    rleFrPoly(rle_instance, cnts, 15, 427, 640);

    // Free the allocated memory
    free(rle_instance->cnts);
    free(rle_instance);

    return 0;
}
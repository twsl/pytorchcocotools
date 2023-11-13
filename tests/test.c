#include <stdlib.h>
#include <math.h>

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

void rleFree(RLE *R)
{
  free(R->cnts);
  R->cnts = 0;
}

char *rleToString(const RLE *R)
{
  /* Similar to LEB128 but using 6 bits/char and ascii chars 48-111. */
  siz i, m = R->m, p = 0;
  long x;
  int more;
  char *s = malloc(sizeof(char) * m * 6);
  for (i = 0; i < m; i++)
  {
    x = (long)R->cnts[i];
    if (i > 2)
      x -= (long)R->cnts[i - 2];
    more = 1;
    while (more)
    {
      char c = x & 0x1f;
      x >>= 5;
      more = (c & 0x10) ? x != -1 : x != 0;
      if (more)
        c |= 0x20;
      c += 48;
      s[p++] = c;
    }
  }
  s[p] = 0;
  return s;
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
  rle_instance->h = 25;
  rle_instance->w = 25;
  rle_instance->m = 10;

  // Define the cnts data
  uint cnts_data[] = {0, 5, 20, 5, 20, 5, 20, 5, 20, 5};

  // Allocate memory for cnts and copy the data
  size_t cnts_size = sizeof(cnts_data) / sizeof(cnts_data[0]);
  rle_instance->cnts = (uint *)malloc(cnts_size * sizeof(uint));
  for (size_t i = 0; i < cnts_size; ++i)
  {
    rle_instance->cnts[i] = cnts_data[i];
  }

  // ... use the rle_instance as needed ...
  char *rle_string = rleToString(rle_instance);

  // Free the allocated memory
  free(rle_instance->cnts);
  free(rle_instance);

  return 0;
}
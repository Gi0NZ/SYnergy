__kernel void vector_add(
    __global const float* a,
    __global const float* b,
    __global float* c,
    unsigned int n,
    unsigned int repeat
) {
    unsigned int i = get_global_id(0);

    if (i < n) {
        float value = a[i] + b[i];

        for (unsigned int r = 0; r < repeat; ++r) {
            value = value + 0.0f;
        }

        c[i] = value;
    }
}
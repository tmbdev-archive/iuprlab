#include <stdio.h>

int test() {
    printf("hello\n");
}

int argmindist_float(int d,int n,float v[d],float data[n][d]) {
    float dists[n];
#pragma omp parallel for shared(dists) schedule(static,50)
    for(int i=0;i<n;i++) {
        float total = 0.0;
        for(int j=0;j<d;j++) {
            float delta = v[j]-data[i][j];
            total += delta*delta;
        }
        dists[i] = total;
    }
    int mi = -1;
    float mv = 1e37;
    for(int i=0;i<n;i++) {
        if(dists[i]>mv) continue;
        mi = i;
        mv = dists[i];
    }
    return mi;
}

int argmindist_double(int d,int n,double v[d],double data[n][d]) {
    double dists[n];
#pragma omp parallel for shared(dists) schedule(static,50)
    for(int i=0;i<n;i++) {
        double total = 0.0;
        for(int j=0;j<d;j++) {
            double delta = v[j]-data[i][j];
            total += delta*delta;
        }
        dists[i] = total;
    }
    int mi = -1;
    double mv = 1e300;
    for(int i=0;i<n;i++) {
        if(dists[i]>mv) continue;
        mi = i;
        mv = dists[i];
    }
    return mi;
}

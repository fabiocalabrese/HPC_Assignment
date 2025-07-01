#include <stdio.h>
#include <stdlib.h>

int main() {
    FILE *seqFile = fopen("log_seq_num.txt", "r");
    FILE *parFile = fopen("log_32x8totalnum.txt", "r");
    FILE *output = fopen("output32x8.txt", "w");
    if (!seqFile || !parFile || !output) {
        printf("Errore nell'aprire i file.\n");
        return 1;
    }

    long long threads_par;
    double tseq_s, tpar_ms;

    // Intestazione sia su schermo che nel file di output
    printf("Threads\t\tTseq(ms)\tTpar(ms)\tSpeedup\t\tThroughput (MPix/s)\n");
    fprintf(output, "Threads\t\tTseq(ms)\tTpar(ms)\tSpeedup\t\tThroughput (MPix/s)\n");

    while (fscanf(seqFile, "%lf", &tseq_s) != EOF &&
           fscanf(parFile, "%lld %lf", &threads_par, &tpar_ms) != EOF) {

        double tseq_ms = tseq_s * 1000.0; // converti secondi in millisecondi
        printf("%f\n", tseq_ms);
        double speedup = tseq_ms / tpar_ms;
        double throughput = threads_par / tpar_ms / 1e6; // in MPix/s

        fprintf(output, "%lld\t\t%.2f\t\t%.2f\t\t%.2f\t\t%.4f\n",
                threads_par, tseq_ms, tpar_ms, speedup, throughput);
           }

    fclose(seqFile);
    fclose(parFile);
    fclose(output);
    return 0;
}

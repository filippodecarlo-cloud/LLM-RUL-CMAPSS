# Guideline for reading the extractor-validation sample

This is the guideline the author followed when labelling the 139 (response, sensor)
pairs in `p3_3_extractor_sample.csv`. The reading was done in Italian; the original
text is reproduced verbatim at the end, and the English below is a translation of it.

## The task

For each case, read the model's response and choose what **the text says** about the
trend of the sensor named. The data are not judged, and neither is whether the model is
right: the comparison with the data is the job of the faithfulness analysis. The chart
shows the values the model had in its prompt, as context only.

## The six labels

| Label | Meaning |
|---|---|
| increase | Values rising, climbing, increasing. |
| decrease | Values falling, decreasing. |
| stable | Only if the text says the sensor is stable, constant, unchanged. |
| unspecified | The sensor is named, but the text does not say whether it rises, falls or stays stable: its level or its variability is described, or it is merely mentioned. |
| absent | The sensor does not appear in the text in any form. |
| ambiguous | It cannot be decided, or two mentions contradict each other. |

The sensor is highlighted wherever its name occurs. Where it does not occur, check
whether the response refers to it in some other way (for example "sensor 12"); if it
does not refer to it at all, the label is absent.

## The four rules

1. **Lists.** If the direction is stated once for a list ("increasing values in s11,
   s12 and s15"), it applies to every sensor in the list.
2. **Consequences.** What counts is the direction attributed to the sensor's values, not
   the direction of the consequence: "increasing s11, indicating a decline in
   performance" is an increase.
3. **Several mentions.** If one mention carries a direction and the others do not, that
   one counts. If two mentions give opposite directions, the case is ambiguous.
4. **Adjectives.** "steady increase", "gradual increase", "consistent rise" are
   increases: the adjective describes the increase, it does not say the sensor is
   stable.

## How the cases were presented

The 139 cases were shown one at a time in a shuffled order (`case` in
`p3_3_author_labels.csv`, key in `p3_3_author_key.csv`), each with the full response, a
chart of the values of that sensor in the prompt and, on request, the whole prompt. The
extractor's verdict was never shown. When the sensor's name did not occur in the
response the screen said so, which is the same test the extractor applies before calling
a pair absent; for those cases the reader's task was to check whether the sensor was
referred to in some other form. The interface recorded the first label given for each
case, which is the one used, and could show the provisional label that preceded this
reading only after a label had been given; it was not opened for any case (column
`revealed`). A first pass over 40 cases was set aside when it showed that the difference
between absent and unspecified was not being applied as intended; the guideline was then
made explicit on that point and on adjectives such as "steady" (rule 4), the notice for a
missing sensor name was added, and all 139 cases were read again.

## Original text (Italian), as shown to the reader

> Per ogni caso leggi la risposta del modello e scegli che cosa **dice il testo**
> sull'andamento del sensore indicato. **Non si giudicano i dati** e non si giudica se il
> modello ha ragione: il confronto con i dati lo fa già l'analisi di fedeltà. Il grafico
> mostra i valori che il modello aveva nel prompt, solo come contesto.
>
> **«Non citato»**: il sensore non compare nel testo in nessuna forma. **«Nominato senza
> direzione»**: compare, ma il testo non dice se sale, scende o resta stabile.
> **«Stabile»** solo se il testo dice che il sensore è stabile, costante, invariato.
>
> Il sensore è evidenziato dove compare. Se non compare, controlla se la risposta ne
> parla in un altro modo (per esempio «sensor 12»): se non ne parla affatto, è «non
> citato».
>
> - **Elenchi.** Se la direzione è detta una volta per un elenco («increasing values in
>   s11, s12 and s15»), vale per ogni sensore dell'elenco.
> - **Conseguenze.** Conta la direzione attribuita ai valori del sensore, non quella della
>   conseguenza: «increasing s11, indicating a decline in performance» è aumenta.
> - **Più menzioni.** Se una menzione ha una direzione e le altre no, vale quella. Se due
>   menzioni danno direzioni opposte, è ambiguo.
> - **Aggettivi.** «steady increase», «gradual increase», «consistent rise» sono aumenti:
>   l'aggettivo descrive l'aumento, non dice che il sensore è stabile.
>
> Le etichette: **Aumenta** (valori in aumento, in salita, crescenti); **Diminuisce**
> (valori in calo, in diminuzione); **Stabile** (valori stabili, costanti, invariati);
> **Nominato senza direzione** (se ne dice il livello, la variabilità, o lo si cita e
> basta); **Non citato** (la risposta non parla di questo sensore); **Ambiguo** (non si
> può decidere, o due menzioni si contraddicono).

:- module(olympics_rules, [
    olympic_advice/4,
    explain_verdict/4
]).

/** <module> Knowledge Base per Predizione Medaglie Olimpiche

    Questo modulo implementa un sistema esperto ibrido. Integra le predizioni
    probabilistiche di un modello Random Forest con una base di conoscenza
    simbolica per fornire analisi contestualizzate e spiegabili.
*/

% --- DIRETTIVE DI SISTEMA ---
% Informano il compilatore che i fatti o le clausole dei seguenti predicati
% possono essere non consecutivi all'interno del file, evitando warning in compilazione.
:- discontiguous historical_elite/3.
:- discontiguous reason/4.


% =============================================================================
% --- 1. FATTI STATICI ---
% =============================================================================

/** historical_elite(?NOC, ?Sport, -Dominance)
    Rappresenta le nazioni leader e il loro peso percentuale nel medagliere storico.
*/
historical_elite('ITA', 'Fencing', 20.12).
historical_elite('USA', 'Swimming', 34.80).
historical_elite('USA', 'Golf', 70.69).
historical_elite('CHN', 'Table Tennis', 45.61).
historical_elite('JAM', 'Athletics', 30.00).
historical_elite('JPN', 'Judo', 38.00).
historical_elite('BRA', 'Football', 25.00).


% =============================================================================
% --- 2. REGOLE AUSILIARIE ---
% =============================================================================

/** is_superpower(+NOC)
    Vero se la nazione possiede una struttura d'élite radicata in più settori (almeno 2).
*/
is_superpower(NOC) :- 
    findall(S, historical_elite(NOC, S, _), L),
    length(L, N), N >= 2.

/** sector_type(+Sport, -Type)
    Definisce la fluidità di un settore (open/closed) in base alla dominanza del leader storico.
    Se la dominanza del leader supera il 40%, il settore viene considerato chiuso (blindato).
*/
sector_type(Sport, closed) :- 
    historical_elite(_, Sport, Dominance), Dominance > 40.0, !.
sector_type(_, open).


% =============================================================================
% --- 3. REGOLE DI INFERENZA PRINCIPALI ---
% =============================================================================

/** olympic_advice(+Probability, +NOC, +Sport, -Advice)
    Predicato principale deputato a generare il verdetto finale testuale.
    Sfrutta l'operatore di Cut (!) per garantire l'esclusività dei rami inferenziali.
    
    @param Probability Valore float da 0 a 1 generato dal modello Random Forest.
    @param NOC Codice del comitato olimpico nazionale dell'atleta (es. 'ITA').
    @param Sport Denominazione della disciplina olimpica analizzata.
    @param Advice Stringa di testo contenente il verdetto predittivo finale.
*/

% Caso A: Eccellenza Storica Confermata (Il modello ML e la KB concordano)
olympic_advice(Prob, NOC, Sport, Advice) :-
    Prob >= 0.70,
    historical_elite(NOC, Sport, Dom),
    atomic_list_concat(['SUCCESSO ATTESO: Dominio storico confermato (Dominanza leader: ', Dom, '%).'], Advice), !.

% Caso B: Tradizione Resiliente (La solidità storica della nazione sostiene un dato statistico incerto)
olympic_advice(Prob, NOC, Sport, "TRADIZIONE COMPETITIVA: La solida storia nazionale sostiene l'atleta nonostante i dati statistici incerti.") :-
    Prob >= 0.45,
    historical_elite(NOC, Sport, _), !.

% Caso C: Nuova Potenza (Exploit in ascesa statistica confinato in un settore fluido e competitivo)
olympic_advice(Prob, NOC, Sport, "NUOVA POTENZA: Trend statistico eccellente in un settore competitivo e fluido.") :-
    Prob >= 0.70,
    \+ historical_elite(NOC, Sport, _),
    sector_type(Sport, open), !.

% Caso D: Exploit Difficile (Il modello statistico è ottimista, ma il settore è storicamente polarizzato)
olympic_advice(Prob, NOC, Sport, "EXPLOIT DIFFICILE: Il modello è ottimista, ma il settore è storicamente blindato da leader dominanti.") :-
    Prob >= 0.70,
    \+ historical_elite(NOC, Sport, _),
    sector_type(Sport, closed), !.

% Caso E: Scommessa (Atleta individualmente promettente ma privo di radici sistemiche nazionali)
olympic_advice(Prob, _, _, "SCOMMESSA: Segnali promettenti, ma contesto privo di radici storiche consolidate.") :-
    Prob >= 0.45, !.

% Caso F: Fallback / Sfida Estrema (Bassa evidenza sia sul fronte statistico che sul fronte storico)
olympic_advice(_, _, _, "SFIDA ESTREMA: Scarsa evidenza statistica e storica per una posizione di podio.").


% =============================================================================
% --- 4. GENERAZIONE MOTIVAZIONI (EXPLAINABLE AI) ---
% =============================================================================

/** reason(+Probability, +NOC, +Sport, -Message)
    Predicato interno che mappa i singoli prerequisiti atomici soddisfatti.
*/
reason(Prob, _, _, 'Solidità statistica del modello ML') :-
    Prob >= 0.70.

reason(_, NOC, Sport, 'Forte tradizione storica nazionale') :-
    historical_elite(NOC, Sport, _).

reason(_, NOC, _, 'Appartenenza a nazione leader (Superpower)') :-
    is_superpower(NOC).

reason(Prob, _, Sport, 'Settore competitivo aperto a nuovi talenti') :-
    Prob >= 0.45,
    sector_type(Sport, open).

reason(Prob, _, _, 'Performance storiche e recenti insufficienti') :-
    Prob < 0.45.


/** explain_verdict(+Probability, +NOC, +Sport, -Reasons)
    Genera l'insieme aggregato e univoco di stringhe motivazionali a supporto della decisione.
    Utilizza findall/3 per accumulare i messaggi e sort/2 per rimuovere i duplicati.
    
    @param Probability Valore float da 0 a 1 generato dal modello statistico.
    @param NOC Codice del comitato olimpico nazionale dell'atleta.
    @param Sport Denominazione della disciplina olimpica.
    @param Reasons Lista contenente l'insieme unico delle spiegazioni estratte.
*/
explain_verdict(Prob, NOC, Sport, Reasons) :-
    findall(M, reason(Prob, NOC, Sport, M), RawReasons),
    sort(RawReasons, UniqueReasons),
    (UniqueReasons = [] -> Reasons = ['Analisi basata su parametri standard'] ; Reasons = UniqueReasons).
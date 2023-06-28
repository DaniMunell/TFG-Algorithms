
%==========================================================================
%
% TRABAJO FIN DE GRADO - Ingeniería Matemática UCM
%
% Daniel Munell Blanco
%
%
%                       =====================
%                       ARTIFICIAL BEE COLONY
%                       =====================
%
%
% El siguiente código está estructurado de forma que cada sección se puede
% ejecutar independientemente del resto sin necesidad de modificar ningún
% parámetro. El código se divide en los siguientes apartados:
%
%   TEST 1 : Elección del Operador de Vecindad
%
%   TEST 2 : Importancia del Parámetro limit
%
%   COMPARACIÓN ALGORITMOS : ABC vs AS vs ACS (común a los tres códigos)
%
%   IMPLEMENTACIÓN DEL ALGORITMO ABC
%
%   FUNCIONES AUXILIARES
%
% Los tres primeros apartados corresponden con lo visto en 
% el capítulo Experiencias Computacionales.
%
%
% NOTA : La comparación de algoritmos lleva 9 minutos por instancia.
%
% ADVERTENCIA : Los archivos de texto de cada instancia deben SOLO 
%               contener las coordenadas de las ciudades dadas en el
%               formato usado por TSPLIB. Es decir, tres columnas:
%                índice de la ciudad, coordenada x, coordenada y
%
%==========================================================================

clear all

%%

% ============================================================
%                           TEST 1
%
%               Elección del Operador de Vecindad
%
% ============================================================


%           _____________________ 
%          |                     |
%          | PROBLEMA : berlin52 |
%          |_____________________|


% Número de ciudades
numCities = 52;

% Tomamos las coordenadas del archivo y calculamos la matriz de distancias
[~, distMatrix] = getCoords(numCities, 'berlin52.tsp');

% Fijamos la longitud óptima de la instancia
opt = 7544.3659;


% --------------- PARÁMETROS ---------------

rng(1)

numBees = 26;                   % Numero de abejas empleadas (= fuentes de alimento = abejas espectadoras)
maxIters = 20000;               % Iteraciones máximas permitidas
limit = numBees*numCities*50;   % Numero de intentos antes de abandonar una fuente
maxTime = Inf;                  % Tiempo máximo permitido

listOperators = ["RS", "RSS", "RRSS", "SWAPMIX"];
dataLength_test1 = zeros(length(listOperators), 4);

numTests = 20;                  % Número de ejecuciones

% ------------------------------------------

for i = 1:length(listOperators)
    neighbourOperator = listOperators(i);  

    [allBestLength, meanBestLength, meanTime] = testAlgorithm(numTests, distMatrix, numBees, maxIters, limit, neighbourOperator, maxTime);
    dataLength_test1(i,:) = [allBestLength, meanBestLength, meanTime, (meanBestLength/opt - 1)*100];
end


%%

% ============================================================
%                           TEST 2
%
%               Importancia del Parámetro limit
%
% ============================================================


%           _____________________ 
%          |                     |
%          | PROBLEMA : berlin52 |
%          |_____________________|


numCities = 52;
[~, distMatrix] = getCoords(numCities, 'berlin52.tsp');
opt = 7544.3659;


% --------------- PARÁMETROS ---------------

rng(2)

numBees = 26;
maxIters = 10000;
neighbourOperator = "SwapMix";
maxTime = Inf;

listLimit = [1/10, 1, 10, 50];

% ------------------------------------------

for i = 1:length(listLimit)
    limit = numBees*numCities*listLimit(i);
    
    [~, ~, bestLengthIter] = ABC(distMatrix, numBees, maxIters, limit, neighbourOperator, maxTime);
    plot(bestLengthIter)
    hold on
end
xlabel("Iteraciones") 
ylabel("Mejor Longitud por Iteración")
yline(opt,'--','Óptimo','LabelHorizontalAlignment', 'left','LabelVerticalAlignment','bottom')
legend("limit = nBees*nCities*" + listLimit)
hold off


%%

% ============================================================
%                 COMPARACIÓN ALGORITMOS
%
%                    ABC vs AS vs ACS
%
% ============================================================


%               EJECUTARLO PRIMERO               < ----------------


% Número de instancias a resolver
numInstances = 4;   

% Posibles valores del parámetro maxTime
listMaxTime = [1, 5, 10, 30];
% Número de posibles valores de maxTime
numTimes = length(listMaxTime); 

% Número de ejecuciones a realizar para cada valor de maxTime
numTests = 10; 

% Vector para almacenar la mejor longitud encontrada en numTests
% ejecuciones, la media de las mejores longitudes y el error relativo
% de la media.
dataLengthABC = zeros(numTimes, 3, numInstances); 

% NOTA : 9 minutos por instancia


%%

%            _____________________ 
%          |                     |
%          | PROBLEMA : berlin52 |
%          |_____________________|


numCities = 52;
[~, distMatrix] = getCoords(numCities, 'berlin52.tsp');
opt = 7544.3659;


% --------------- PARÁMETROS ---------------

rng(10)

numBees = ceil(numCities/2);
maxIters = 10^6;
limit = numBees*numCities*50;
neighbourOperator = "SwapMix";

% ------------------------------------------

for i = 1:numTimes
    maxTime = listMaxTime(i);  

    [allBestLength, meanBestLength, meanTime] = testAlgorithm(numTests, distMatrix, numBees, maxIters, limit, neighbourOperator, maxTime);
    dataLengthABC(i,:,1) = [allBestLength, meanBestLength, (meanBestLength/opt - 1)*100];
end


%%

%           _____________________ 
%          |                     |
%          |  PROBLEMA : kroA100 |
%          |_____________________|


numCities = 100;
[coords, distMatrix] = getCoords(numCities, 'kroA100.tsp');
opt = 21285.44;


% --------------- PARÁMETROS ---------------

rng(10)

numBees = ceil(numCities/2);
maxIters = 10^6;
limit = numBees*numCities*50;
neighbourOperator = "SwapMix";

% ------------------------------------------

for i = 1:numTimes
    maxTime = listMaxTime(i);  

    [allBestLength, meanBestLength, meanTime] = testAlgorithm(numTests, distMatrix, numBees, maxIters, limit, neighbourOperator, maxTime);
    dataLengthABC(i,:,2) = [allBestLength, meanBestLength, (meanBestLength/opt - 1)*100];
end


%%

%           _____________________ 
%          |                     |
%          |   PROBLEMA : d198   |
%          |_____________________|


numCities = 198;
[~, distMatrix] = getCoords(numCities, 'd198.tsp');
opt = 15780;


% --------------- PARÁMETROS ---------------

rng(10)

numBees = ceil(numCities/2);
maxIters = 10^6;
limit = numBees*numCities*50;
neighbourOperator = "SwapMix";

% ------------------------------------------

for i = 1:numTimes
    maxTime = listMaxTime(i);  

    [allBestLength, meanBestLength, meanTime] = testAlgorithm(numTests, distMatrix, numBees, maxIters, limit, neighbourOperator, maxTime);
    dataLengthABC(i,:,3) = [allBestLength, meanBestLength, (meanBestLength/opt - 1)*100];
end


%%

%           _____________________ 
%          |                     |
%          |  PROBLEMA : lin318  |
%          |_____________________|


numCities = 318;
[~, distMatrix] = getCoords(numCities, 'lin318.tsp');
opt = 42029;


% --------------- PARÁMETROS ---------------

rng(10)

numBees = ceil(numCities/2);
maxIters = 10^6;
limit = numBees*numCities*50;
neighbourOperator = "SwapMix";

% ------------------------------------------

for i = 1:numTimes
    maxTime = listMaxTime(i);  

    [allBestLength, meanBestLength, meanTime] = testAlgorithm(numTests, distMatrix, numBees, maxIters, limit, neighbourOperator, maxTime);
    dataLengthABC(i,:,4) = [allBestLength, meanBestLength, (meanBestLength/opt - 1)*100];
end


%%

% ============================================================
%                         ALGORITMO
% ============================================================



function [bestLength, bestSolution, bestLengthIter] = ABC(distMatrix, numBees, maxIters, limit, neighbourOperator, maxTime)
% =======================================================
% Aplica el algoritmo ABC (Artificial Bee Colony)
%   
%   INPUTS:
%       
%       distMatrix : matriz de distancias entre ciudades
%       numBees : número de abejas empleadas
%           (= número de abejas espectadoras = fuentes de alimento)
%       maxIters : máximo número de iteraciones permitidas
%       limit : máximo número de intentos para mejorar una solución    
%       neighbourOperator : operador de vecindario a utilizar
%           ("RS", "RSS", "RRSS", "SWAPMIX")
%       maxTime : máximo tiempo de ejecución 
%
%   OUTPUTS:
%
%       bestLength : longitud del mejor tour encontrado
%       bestSolution : permutación del mejor tour encontrado
%
%   NOTA : La función fitness usada es la longitud del 
%           tour dada su mayor eficiencia.
%
% =======================================================


    % -------------------- INICIALIZACIÓN --------------------
    
    numCities = size(distMatrix, 1);    % Numero de ciudades

    % Inicialización de las soluciones (fuentes de alimento)
    foodSources = zeros(numBees, numCities);
    for i = 1:numBees
        foodSources(i, :) = randperm(numCities);
    end
    
    % Evaluación del fitness de cada solución
    fitness = zeros(1, numBees);
    for i = 1:numBees
        fitness(i) = getTourLength(foodSources(i, :), distMatrix);
    end
    
    % Se busca la mejor solución inicial   
    [bestLength, bestIndex] = min(fitness);                       % <-----
    bestSolution = foodSources(bestIndex, :);

    bestLengthIter = zeros(1, maxIters); 
    
    % Inicialización del contador de intentos de mejora
    triesCounter = zeros(1, numBees);
    
    % Para mayor eficiencia usamos valores numéricos
    if neighbourOperator == "RS"
        neighbourOperator = 1;
    elseif neighbourOperator == "RSS"
        neighbourOperator = 2;
    elseif neighbourOperator == "RRSS"
        neighbourOperator = 3;
    else % SwapMix
        neighbourOperator = 4;
    end        

    % --------------------------------------------------------       

    startTime = tic;

    % Bucle Principal
    for iter = 1:maxIters
        
        if toc(startTime) > maxTime
            return
        end
    
        % --------------------- FASE ABEJAS EMPLEADAS ---------------------
    
        for i = 1:numBees
    
            % Se calcula una solución candidata usando el operador dado
            sequence = foodSources(i, :);
            if neighbourOperator == 4
                newSolution = swapMix(sequence);
            elseif neighbourOperator == 3
                newSolution = RRSS(sequence);
            elseif neighbourOperator == 2
                newSolution = RSS(sequence);
            else
                newSolution = RS(sequence);
            end
            newSolutionLength = getTourLength(newSolution, distMatrix);
            
            % Si la solución candidata es mejor se almacena, en caso
            % contrario se aumenta el contador de intentos de mejora
            if newSolutionLength < fitness(i)                    
                foodSources(i, :) = newSolution;
                fitness(i) = newSolutionLength;
    
                triesCounter(i) = 0;
            else
                triesCounter(i) = triesCounter(i) + 1;
            end
    
        end
    
        % -----------------------------------------------------------------
    
    
    
        % ------------------- FASE ABEJAS ESPECTADORAS --------------------
            
        % Calculamos la probabilidad que las espectadoras escojan cada solucion
        probs = fitness / sum(fitness);
        
        
        for i = 1:numBees
    
            % Se escoge una abeja empleada (su fuente de alimento) en funcion
            % de la probabilidad anterior
            beeIndex = chooseBee(probs);
            
            % Se calcula una solución candidata usando el operador dado
            sequence = foodSources(beeIndex, :);
            if neighbourOperator == 4
                newSolution = swapMix(sequence);
            elseif neighbourOperator == 3
                newSolution = RRSS(sequence);
            elseif neighbourOperator == 2
                newSolution = RSS(sequence);
            else
                newSolution = RS(sequence);
            end
            newSolutionLength = getTourLength(newSolution, distMatrix);
            
            % Si la solución candidata es mejor se almacena, en caso
            % contrario se aumenta el contador de intentos de mejora
            if newSolutionLength < fitness(beeIndex)             
                foodSources(beeIndex, :) = newSolution;
                fitness(beeIndex) = newSolutionLength;
    
                triesCounter(beeIndex) = 0;
            else
                triesCounter(beeIndex) = triesCounter(beeIndex) + 1;
            end
    
        end
    
    
        % Se busca la mejor solución hasta el momento
        [minLength, minIndex] = min(fitness);            
        if minLength < bestLength                      
            bestLength = minLength;
            bestSolution = foodSources(minIndex, :);
        end

        bestLengthIter(iter) = minLength;
    
        % -----------------------------------------------------------------
    
       
    
        % ------------------- FASE ABEJAS EXPLORADORAS --------------------
    
        for i = 1:numBees
    
            % Comprueba si la solución no ha podido ser mejorada
            if triesCounter(i) > limit
    
                % Se busca una nueva solución aleatoria
                newSolution = randperm(numCities);
                foodSources(i, :) = newSolution;
                fitness(i) = getTourLength(newSolution, distMatrix);
                triesCounter(i) = 0;
            end
        end
    
        % -----------------------------------------------------------------
    
    end

%     % Muestra la longitud de la mejor solución encontrada
%     fprintf('---------\n')
%     fprintf('Mejor Longitud: %s\n', num2str(bestLength));
    
end




%%



% ============================================================
%                         FUNCIONES
% ============================================================


function newSequence = RS(sequence)
% =======================================================
% Aplica el operador de vecindario RS
% (Random Swap)
%   
%   INPUTS:
%       
%       sequence : permutación del tour a transformar
%
%   OUTPUTS:
%
%       newSequence : nueva permutación del tour
%
% =======================================================
    n = length(sequence);

    citiesToSwap = randperm(n, 2);
            
    newSequence = sequence;
    temp = newSequence(citiesToSwap(1));
    newSequence(citiesToSwap(1)) = newSequence(citiesToSwap(2));
    newSequence(citiesToSwap(2)) = temp;
end


% ------------------------------------------------------------


function newSequence = RSS(sequence)
% =======================================================
% Aplica el operador de vecindario RSS
% (Random Swap of Subsequences)
%   
%   INPUTS:
%       
%       sequence : permutación del tour a transformar
%
%   OUTPUTS:
%
%       newSequence : nueva permutación del tour
%
% =======================================================
    n = length(sequence);
    
    head1 = randi(n-1); head2 = randi([head1 + 1, n]);
    last1 = randi([head1, head2 - 1]); last2 = randi([head2, n]);
    
    len1 = last1 - head1; len2 = last2 - head2;

    newSequence = sequence;
    newSequence(head1 : head1+len2) = sequence(head2:last2);
    newSequence(head1+len2+1 : last2-len1-1) = sequence(last1+1:head2-1);
    newSequence(last2-len1 : last2) = sequence(head1:last1);

end


% ------------------------------------------------------------


function newSequence = RRSS(sequence)
% =======================================================
% Aplica el operador de vecindario RRSS
% (Random Reversing Swap of Subsequences)
%   
%   INPUTS:
%       
%       sequence : permutación del tour a transformar
%
%   OUTPUTS:
%
%       newSequence : nueva permutación del tour
%
% =======================================================
    n = length(sequence);
    
    head1 = randi(n-1); head2 = randi([head1 + 1, n]);
    last1 = randi([head1, head2 - 1]); last2 = randi([head2, n]);
    coin1 = randi([0, 1]); coin2 = randi([0, 1]);
    
    len1 = last1 - head1; len2 = last2 - head2;   

    newSequence = sequence;
    if coin1 == 0
        newSequence(head1 : head1+len2) = sequence(head2:last2);
    else
        newSequence(head1 : head1+len2) = sequence(last2:-1:head2);
    end
    newSequence(head1+len2+1 : last2-len1-1) = sequence(last1+1:head2-1);
    if coin2 == 0
        newSequence(last2-len1 : last2) = sequence(head1:last1);
    else
        newSequence(last2-len1 : last2) = sequence(last1:-1:head1);
    end
end


% ------------------------------------------------------------


function newSequence = swapMix(sequence)
% =======================================================
% Aplica con igual probabilidad los operadores RS, RSS, RRSS
%   
%   INPUTS:
%       
%       sequence : permutación del tour a transformar
%
%   OUTPUTS:
%
%       newSequence : nueva permutación del tour
%
% =======================================================
    integer = randi(3);

    if integer == 1
        newSequence = RS(sequence);
    elseif integer == 2
        newSequence = RSS(sequence);
    else
        newSequence = RRSS(sequence);
    end
end


% ------------------------------------------------------------


function tourLength = getTourLength(tour, distMatrix)
% =======================================================
% Calcula la longitud de un tour dado.
%   
%   INPUTS:
%       
%       tour : permutación del tour
%       distMatrix : matriz de distancias entre ciudades
%
%   OUTPUTS:
%
%       tourLength : longitud del tour
%
% =======================================================
    numCities = length(tour);

    tourLength = 0;
    for i = 1:numCities-1
        tourLength = tourLength + distMatrix(tour(i), tour(i+1));
    end
    tourLength = tourLength + distMatrix(tour(numCities), tour(1));
end


% ------------------------------------------------------------


function selectedIndex = chooseBee(probs)
% =======================================================
% Elige una abeja en función de las probabilidades dadas
%   
%   INPUTS:
%
%       probs : vector de probabilidades (debe sumar 1)
%
%   OUTPUTS:
%
%       selectedIndex : índice de la abeja elegida
%
% =======================================================

    numCities = length(probs);

    r = rand();
    cumSums = cumsum(probs);
    
    for selectedIndex = 1:numCities
        if r <= cumSums(selectedIndex)
            return
        end
    end
end


% ------------------------------------------------------------


function [allBestLength, meanBestLength, meanTime] = testAlgorithm(numTests, distMatrix, numBees, maxIters, limit, neighbourOperator, maxTime)
% =======================================================
% Ejecuta el algoritmo ABC tantas veces como se indique
% y calcula la longitud de tour media obtenida
%   
%   INPUTS:
%
%       numTests : número de ejecuciones del algoritmo
%       ... : veáse parámetros de la función ABC
%
%   OUTPUTS:
%       
%       allBestLength : mejor longitud encontrada
%       meanBestLength : longitud de tour media
%       meanTime : tiempo de ejecución medio
%
% =======================================================

    meanBestLength = 0;
    allBestLength = Inf;
    meanTime = 0;

    for tests = 1:numTests
        tic
        bestLength = ABC(distMatrix, numBees, maxIters, limit, neighbourOperator, maxTime);
        execTime = toc;

        if bestLength < allBestLength
            allBestLength = bestLength;
        end

        meanBestLength = meanBestLength + bestLength;
        meanTime = meanTime + execTime;
    end
    meanBestLength = meanBestLength / numTests;
    meanTime = meanTime / numTests;
end


% ------------------------------------------------------------


function [coords, distMatrix] = getCoords(numCities, fileName)
% =======================================================
% Almacena las coordenadas de un archivo dado y calcula
% la matriz de distancias
%   
%   INPUTS:
%
%       numCities : numero de ciudades del problema
%       fileName : nombre del archivo
%
%   OUTPUTS:
%       
%       coords : vector de coordenadas x,y
%       distMatrix : matriz de distancias
%
% ADVERTENCIA : Los archivos de texto de cada instancia deben SOLO 
%               contener las coordenadas de las ciudades dadas en el
%               formato usado por TSPLIB. Es decir, tres columnas:
%               índice de la ciudad, coordenada x, coordenada y
%
% ======================================================= 

    dataFile = fopen(fileName,'r');
    
    coords = fscanf(dataFile, '%d %f %f', [3 numCities])';
    coords = coords(:,2:3);
    
    distMatrix = squareform(pdist(coords));

end
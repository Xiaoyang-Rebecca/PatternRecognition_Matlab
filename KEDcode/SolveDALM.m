% Copyright ©2010. The Regents of the University of California (Regents). 
% All Rights Reserved. Contact The Office of Technology Licensing, 
% UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, 
% (510) 643-7201, for commercial licensing opportunities.

% Authors: Arvind Ganesh, Allen Y. Yang, Zihan Zhou.
% Contact: Allen Y. Yang, Department of EECS, University of California,
% Berkeley. <yang@eecs.berkeley.edu>

% IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, 
% SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, 
% ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF 
% REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

% REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED
% TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
% PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, 
% PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO 
% PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

%% This function is modified from Matlab code proximal_gradient_bp

function [x, nIter] = SolveDALM(A, b, varargin)

STOPPING_GROUND_TRUTH = -1;
STOPPING_DUALITY_GAP = 1;
STOPPING_SPARSE_SUPPORT = 2;
STOPPING_OBJECTIVE_VALUE = 3;
STOPPING_SUBGRADIENT = 4;
STOPPING_INCREMENTS = 5 ;
STOPPING_DEFAULT = STOPPING_INCREMENTS;

stoppingCriterion = STOPPING_DEFAULT;

tol = 1e-3;
lambda = 1e-2;
maxIter = 5000;
VERBOSE = 0;

% Parse the optional inputs.
if (mod(length(varargin), 2) ~= 0 ),
    error(['Extra Parameters passed to the function ''' mfilename ''' lambdast be passed in pairs.']);
end
parameterCount = length(varargin)/2;

for parameterIndex = 1:parameterCount,
    parameterName = varargin{parameterIndex*2 - 1};
    parameterValue = varargin{parameterIndex*2};
    switch lower(parameterName)
        case 'stoppingcriterion'
            stoppingCriterion = parameterValue;
        case 'groundtruth'
            xG = parameterValue;
        case 'tolerance'
            tol = parameterValue;
        case 'lambda'
            lambda = parameterValue;
        case 'maxiteration'
            maxIter = parameterValue;
        otherwise
            error(['The parameter ''' parameterName ''' is not recognized by the function ''' mfilename '''.']);
    end
end
clear varargin

[m,n] = size(A);

beta = norm(b,1)/m;
betaInv = 1/beta ;

G = A * A' + eye(m) * lambda / beta;
invG = inv(G);
A_invG_b = A' * (invG * b);

nIter = 0 ;

if VERBOSE
    disp(['beta is: ' num2str(beta)]);
end

y = zeros(m,1);
x = zeros(n,1);    
z = zeros(m+n,1);

converged_main = 0 ;

temp = A' * y;
f = norm(x,1);
while ~converged_main
    
    nIter = nIter + 1 ;  
    
    x_old = x;
    
    %update z
    temp1 = temp + x * betaInv;
    z = sign(temp1) .* min(1,abs(temp1));
    
    %compute A' * y    
    temp = A' * (invG * (A * (z - x * betaInv))) + A_invG_b * betaInv; 
    
    %update x
    x = x - beta * (z - temp);
    
    if VERBOSE && mod(nIter, 50) == 0
    
        disp(['Iteration ' num2str(nIter)]) ;
        disp(norm(x-x_old)/norm(x_old));
        
        figure(1);
        subplot(2,1,1);
        plot(x);
        title('x');
        subplot(2,1,2);
        plot(z);
        title('z');
        pause;
    
    end
    
    switch stoppingCriterion
        case STOPPING_GROUND_TRUTH
            if norm(xG-x) < tol
                converged_main = 1 ;
            end
        case STOPPING_SUBGRADIENT
            error('Duality gap is not a valid stopping criterion for ALM.');
        case STOPPING_SPARSE_SUPPORT
            % compute the stopping criterion based on the change
            % of the number of non-zero components of the estimate
            nz_x_prev = nz_x;
            nz_x = (abs(x)>eps*10);
            num_nz_x = sum(nz_x(:));
            num_changes_active = (sum(nz_x(:)~=nz_x_prev(:)));
            if num_nz_x >= 1
                criterionActiveSet = num_changes_active / num_nz_x;
                converged_main = ~(criterionActiveSet > tol);
            end
        case STOPPING_OBJECTIVE_VALUE
            % compute the stopping criterion based on the relative
            % variation of the objective function.
            prev_f = f;
            f = norm(x,1);
            criterionObjective = abs(f-prev_f)/(prev_f);
            converged_main =  ~(criterionObjective > tol);
        case STOPPING_DUALITY_GAP
            error('Duality gap is not a valid stopping criterion for ALM.');
        case STOPPING_INCREMENTS
            if norm(x_old - x) < tol*norm(x_old)
                converged_main = 1 ;
            end
        otherwise
            error('Undefined stopping criterion.');
    end
    
    if ~converged_main && norm(x_old-x)<100*eps
        if VERBOSE
            disp('The iteration is stuck.') ;
        end
        converged_main = 1 ;
    end
    
    if ~converged_main && nIter >= maxIter
        if VERBOSE
            disp('Maximum Iterations Reached') ;
        end
        converged_main = 1 ;
    end
    
end
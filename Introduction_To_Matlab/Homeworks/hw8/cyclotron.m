function [E,n] = cyclotron (V)
    m = 3.344e-27;           % mass of deuteron [kg]             
    q = 1.603e-19;           % charge of deuteron [c]
    B = 1.600;               % magnetic field strength [t]
    z = m*V/(q*B^2);         % initial trajectory radius squared [m^2]
    n = 0;                   % pass number
    d = 2;                   % diameter/radius, sign alternates
    r = sqrt(z);             % initial radius of curvature 
    x = -r/2.0;              % initial x-coordinate

    while x>=-0.5            % while deuteron has not emerged 
        x = x+d*r;           % spiral to next x-coordinate ;
        n = n+1;             % update pass number
        d = -d;              % ... and direction
        r = sqrt(r^2+2*z);   % calculate new radius in case needed
    end
    E = V*n*1e-6;            % determine energy  
end
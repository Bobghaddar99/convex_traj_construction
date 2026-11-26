% Auto-generated driver: paper Section 5.2 DWA on the Figure 9 map.
% Everything from "function [u,trajDB]=DynamicWindowApproach" downward is copied verbatim
% from AtsushiSakai/MATLABRobotics DynamicWindowApproachSample.m (Modified BSD).
% Only this main function differs from the original sample: Table 3 parameters, the
% Diagonal-A* path points used as successive goals, CSV logging, no animation.
function [] = dwa_fig9(outfile, evalParam, obstacleR, reach)
if nargin < 2, evalParam=[0.1,0.2,0.1,3.0]; end   % sample default [heading,dist,velocity,predictDT]
if nargin < 3, obstacleR=0.5; end                  % sample default
if nargin < 4, reach=0.5; end                      % sample goal test: norm(x-goal)<0.5
obstacle=[0.5 3.5;0.5 4.5;1.5 4.5;1.5 5.5;2.5 0.5;2.5 3.5;2.5 8.5;3.5 3.5;3.5 4.5;3.5 7.5;3.5 9.5;4.5 0.5;4.5 2.5;4.5 6.5;5.5 1.5;5.5 5.5;5.5 7.5;5.5 9.5;6.5 6.5;6.5 7.5;7.5 2.5;7.5 5.5;7.5 7.5;8.5 3.5;8.5 4.5;8.5 7.5;9.5 0.5;9.5 2.5;9.5 3.5;9.5 4.5;9.5 8.5];
wp=[0.5 0.5;1.5 1.5;1.5 2.5;1.5 3.5;2.5 4.5;3.5 5.5;3.5 6.5;4.5 7.5;5.5 8.5;6.5 8.5;7.5 8.5;8.5 8.5];
global dt; dt=0.1;                                 % Table 3: search time step 0.1 s
% Table 3: [max v 2 m/s, max w 0.35 rad/s, max acc 0.5 m/s^2, max dw 0.87 rad/s^2, v res 0.02, w res 0.017]
Kinematic=[2.0,0.35,0.5,0.87,0.02,0.017];
yaw0=atan2(wp(2,2)-wp(1,2), wp(2,1)-wp(1,1));
x=[wp(1,1) wp(1,2) yaw0 0 0]';
k=2; log=[];
for i=1:5000                                       % Table 3: maximum window steps
    [u,traj]=DynamicWindowApproach(x,Kinematic,wp(k,:),evalParam,obstacle,obstacleR);
    x=f(x,u);
    log=[log; i*dt x' k-1];
    if norm(x(1:2)-wp(k,:)')<reach
        k=k+1;
        if k>size(wp,1), break; end
    end
end
fid=fopen(outfile,'w'); fprintf(fid, '%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%d\n', log'); fclose(fid);
printf('reached %d/%d path points in %d steps (%.1f s)\n', k-2, size(wp,1)-1, i, i*dt);

function [u,trajDB]=DynamicWindowApproach(x,model,goal,evalParam,ob,R)
%DWAによる入力値の計算をする関数

%Dynamic Window[vmin,vmax,ωmin,ωmax]の作成
Vr=CalcDynamicWindow(x,model);
%評価関数の計算
[evalDB,trajDB]=Evaluation(x,Vr,goal,ob,R,model,evalParam);

if isempty(evalDB)
    disp('no path to goal!!');
    u=[0;0];return;
end

%各評価関数の正規化
evalDB=NormalizeEval(evalDB);

%最終評価値の計算
feval=[];
for id=1:length(evalDB(:,1))
    feval=[feval;evalParam(1:3)*evalDB(id,3:5)'];
end
evalDB=[evalDB feval];

[maxv,ind]=max(feval);%最も評価値が大きい入力値のインデックスを計算
u=evalDB(ind,1:2)';%評価値が高い入力値を返す

function [evalDB,trajDB]=Evaluation(x,Vr,goal,ob,R,model,evalParam)
%各パスに対して評価値を計算する関数
evalDB=[];
trajDB=[];

for vt=Vr(1):model(5):Vr(2)
    for ot=Vr(3):model(6):Vr(4)
        %軌跡の推定
        [xt,traj]=GenerateTrajectory(x,vt,ot,evalParam(4),model);
        %各評価関数の計算
        heading=CalcHeadingEval(xt,goal);
        dist=CalcDistEval(xt,ob,R);
        vel=abs(vt);
        
        evalDB=[evalDB;[vt ot heading dist vel]];
        trajDB=[trajDB;traj];     
    end
end

function EvalDB=NormalizeEval(EvalDB)
%評価値を正規化する関数
if sum(EvalDB(:,3))~=0
    EvalDB(:,3)=EvalDB(:,3)/sum(EvalDB(:,3));
end
if sum(EvalDB(:,4))~=0
    EvalDB(:,4)=EvalDB(:,4)/sum(EvalDB(:,4));
end
if sum(EvalDB(:,5))~=0
    EvalDB(:,5)=EvalDB(:,5)/sum(EvalDB(:,5));
end

function [x,traj]=GenerateTrajectory(x,vt,ot,evaldt,model)
%軌跡データを作成する関数
global dt;
time=0;
u=[vt;ot];%入力値
traj=x;%軌跡データ
while time<=evaldt
    time=time+dt;%シミュレーション時間の更新
    x=f(x,u);%運動モデルによる推移
    traj=[traj x];
end

function stopDist=CalcBreakingDist(vel,model)
%現在の速度から力学モデルに従って制動距離を計算する関数
global dt;
stopDist=0;
while vel>0
    stopDist=stopDist+vel*dt;%制動距離の計算
    vel=vel-model(3)*dt;%最高原則
end

function dist=CalcDistEval(x,ob,R)
%障害物との距離評価値を計算する関数

dist=2;
for io=1:length(ob(:,1))
    disttmp=norm(ob(io,:)-x(1:2)')-R;%パスの位置と障害物とのノルム誤差を計算
    if dist>disttmp%最小値を見つける
        dist=disttmp;
    end
end

function heading=CalcHeadingEval(x,goal)
%headingの評価関数を計算する関数

theta=toDegree(x(3));%ロボットの方位
goalTheta=toDegree(atan2(goal(2)-x(2),goal(1)-x(1)));%ゴールの方位

if goalTheta>theta
    targetTheta=goalTheta-theta;%ゴールまでの方位差分[deg]
else
    targetTheta=theta-goalTheta;%ゴールまでの方位差分[deg]
end

heading=180-targetTheta;

function Vr=CalcDynamicWindow(x,model)
%モデルと現在の状態からDyamicWindowを計算
global dt;
%車両モデルによるWindow
Vs=[0 model(1) -model(2) model(2)];

%運動モデルによるWindow
Vd=[x(4)-model(3)*dt x(4)+model(3)*dt x(5)-model(4)*dt x(5)+model(4)*dt];

%最終的なDynamic Windowの計算
Vtmp=[Vs;Vd];
Vr=[max(Vtmp(:,1)) min(Vtmp(:,2)) max(Vtmp(:,3)) min(Vtmp(:,4))];
%[vmin,vmax,ωmin,ωmax]

function x = f(x, u)
% Motion Model
global dt;
 
F = [1 0 0 0 0
     0 1 0 0 0
     0 0 1 0 0
     0 0 0 0 0
     0 0 0 0 0];
 
B = [dt*cos(x(3)) 0
    dt*sin(x(3)) 0
    0 dt
    1 0
    0 1];

x= F*x+B*u;

function radian = toRadian(degree)
% degree to radian
radian = degree/180*pi;

function degree = toDegree(radian)
% radian to degree
degree = radian/pi*180;
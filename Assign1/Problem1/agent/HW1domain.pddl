(define (domain grid_world ) 
(:requirements :strips :typing) 
(:types car
agent - car
gridcell
) 
(:predicates (at ?pt1 - gridcell ?car - car) 
(up_next ?pt1 - gridcell ?pt2 - gridcell) 
(down_next ?pt1 - gridcell ?pt2 - gridcell) 
(forward_next ?pt1 - gridcell ?pt2 - gridcell) 
(blocked ?pt1 - gridcell) 
) 
(:action UP
:parameters ( ?from - gridcell ?to - gridcell) 
:precondition (and (at ?from agent1) (up_next ?from ?to) (not (blocked ?to)))
:effect (and (at ?to agent1) (not (at ?from agent1)))
) 
(:action DOWN
:parameters ( ?from - gridcell ?to - gridcell) 
:precondition (and (at ?from agent1) (down_next ?from ?to) (not (blocked ?to)))
:effect (and (at ?to agent1) (not (at ?from agent1)))
) 
(:action FORWARD
:parameters ( ?from - gridcell ?to - gridcell) 
:precondition (and (at ?from agent1) (forward_next ?from ?to) (not (blocked ?to)))
:effect (and (at ?to agent1) (not (at ?from agent1)))
) 
) 

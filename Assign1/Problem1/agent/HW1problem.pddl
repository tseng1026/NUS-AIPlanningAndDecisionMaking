(define (problem parking) 
 (:domain grid_world) 
(:objects agent1  - agent
 pt0pt0 pt0pt1 pt0pt2 pt1pt0 pt1pt1 pt1pt2 pt2pt0 pt2pt1 pt2pt2 pt3pt0 pt3pt1 pt3pt2 pt4pt0 pt4pt1 pt4pt2  - gridcell
 ) 
(:init (at pt4pt2 agent1) (blocked pt1pt0) (at pt1pt0 0) (blocked pt1pt2) (at pt1pt2 4) (blocked pt3pt0) (at pt3pt0 1) (blocked pt0pt1) (at pt0pt1 3) (blocked pt0pt2) (at pt0pt2 5) (blocked pt4pt1) (at pt4pt1 2) (down_next pt1pt0 pt0pt1) (forward_next pt1pt0 pt0pt0) (up_next pt1pt1 pt0pt0) (down_next pt1pt1 pt0pt2) (forward_next pt1pt1 pt0pt1) (up_next pt1pt2 pt0pt1) (forward_next pt1pt2 pt0pt2) (down_next pt2pt0 pt1pt1) (forward_next pt2pt0 pt1pt0) (up_next pt2pt1 pt1pt0) (down_next pt2pt1 pt1pt2) (forward_next pt2pt1 pt1pt1) (up_next pt2pt2 pt1pt1) (forward_next pt2pt2 pt1pt2) (down_next pt3pt0 pt2pt1) (forward_next pt3pt0 pt2pt0) (up_next pt3pt1 pt2pt0) (down_next pt3pt1 pt2pt2) (forward_next pt3pt1 pt2pt1) (up_next pt3pt2 pt2pt1) (forward_next pt3pt2 pt2pt2) (down_next pt4pt0 pt3pt1) (forward_next pt4pt0 pt3pt0) (up_next pt4pt1 pt3pt0) (down_next pt4pt1 pt3pt2) (forward_next pt4pt1 pt3pt1) (up_next pt4pt2 pt3pt1) (forward_next pt4pt2 pt3pt2)) 
(:goal (at pt0pt0 agent1)) 
) 

(define (domain living-room-placement)
  (:requirements :strips :typing)
  (:types object region)
  (:predicates (available ?o - object) (holding ?o - object)
               (hand-empty) (on ?o - object ?r - region))
  (:action pick
    :parameters (?o - object)
    :precondition (and (available ?o) (hand-empty))
    :effect (and (holding ?o) (not (available ?o)) (not (hand-empty))))
  (:action place
    :parameters (?o - object ?r - region)
    :precondition (holding ?o)
    :effect (and (on ?o ?r) (hand-empty) (not (holding ?o))))
)

(define (domain vlm-tamp-workshop)
  (:requirements :strips :negative-preconditions :action-costs)
  (:predicates
    (Movable ?o) (Target ?o) (CompatibleFastener ?o) (CompatibleDriver ?o)
    (Region ?r) (Storage ?r) (Destination ?x)
    (At ?o ?x) (Open ?r) (Inspected ?r) (Accessible ?x)
    (HandEmpty) (Holding ?o) (Inserted ?fastener ?target)
    (Fastened ?tool ?fastener ?target))
  (:functions (total-cost))

  (:action inspect
    :parameters (?region)
    :precondition (and (Storage ?region) (HandEmpty))
    :effect (and (Open ?region) (Inspected ?region) (Accessible ?region)
                 (increase (total-cost) 1)))

  (:action pick
    :parameters (?object ?source)
    :precondition (and (Movable ?object) (At ?object ?source)
                       (Accessible ?source) (HandEmpty))
    :effect (and (Holding ?object) (not (HandEmpty))
                 (not (At ?object ?source)) (increase (total-cost) 1)))

  (:action place
    :parameters (?object ?destination)
    :precondition (and (Movable ?object) (Holding ?object)
                       (Destination ?destination))
    :effect (and (At ?object ?destination) (HandEmpty)
                 (not (Holding ?object)) (increase (total-cost) 1)))

  (:action insert
    :parameters (?fastener ?target)
    :precondition (and (Movable ?fastener) (CompatibleFastener ?fastener) (Target ?target)
                       (Holding ?fastener))
    :effect (and (Inserted ?fastener ?target) (HandEmpty)
                 (not (Holding ?fastener)) (increase (total-cost) 1)))

  (:action fasten
    :parameters (?tool ?fastener ?target)
    :precondition (and (Movable ?tool) (CompatibleDriver ?tool) (Holding ?tool)
                       (Inserted ?fastener ?target) (Target ?target))
    :effect (and (Fastened ?tool ?fastener ?target)
                 (increase (total-cost) 1))))

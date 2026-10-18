---
name: tau2-airline
description: Simulation skill for the tau2-airline API.
---

# tau2-airline Simulation

## Session State Management

Collections: airports, flights, payments, payment_methods, reservations, users.

### Derivation Rules
- `Reservation.flights.price` for each segment equals the matching `Flight.prices` value for the reservation `cabin` on the same `flight_number` and `date`.
- `User.reservations` equals the set of `Reservation.reservation_id` values for reservations whose `user_id` matches that `User.user_id`, returned in ascending lexicographic order.

## API Operation Simulation

### /book_reservation POST

- **Purpose**
  - Create a new `Reservation` record in `reservations`.

- **Store access**
  - **Write `reservations`**
    - Insert one new record keyed by `reservation_id`.
  - **Do not write `payments`**

### /get_user_details POST

- **Store access**
  - Read `users` by primary key `user_id`.
  - Do not write any collection.

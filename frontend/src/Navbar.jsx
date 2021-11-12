import React,{useState} from "react";
import { NavLink } from "react-router-dom";
import logo from "../src/Image/ailogo1.png";
import { useHistory } from "react-router";
//redux import
import { userLogout } from "./redux/actions/userActions";
import { useDispatch, useSelector } from "react-redux";
import fire from "./fire";

const Navbar = () => {

  const dispatch = useDispatch();
  const history = useHistory()
  const { userDetails } = useSelector((state) => state.userLogin);
  const [numOfcoins, SetNoOfCoins] = useState("")
  console.log(userDetails);


  function logoutHandler() {
    dispatch(userLogout())
    if (dispatch) {
      history.go(0)
    }

    setTimeout(
      function () {
        history.go(0)
      },
      1000
    )
  }

  return (
    <>
      <div className="container-fluid1 nav_bg1">
        <div className="row">
          <div className="col-10 mx-auto">
            <nav className="navbar navbar-expand-lg navbar-light bg-light">
              <NavLink exact className="navbar-brand font-weight-bold " to="/">
              <img className="navlogo" src={logo} />
                IEMAIWRITER
              </NavLink>
              <button
                className="navbar-toggler"
                type="button"
                data-toggle="collapse"
                data-target="#navbarSupportedContent"
                aria-controls="navbarSupportedContent"
                aria-expanded="false"
                aria-label="Toggle navigation"
              >
                <span className="navbar-toggler-icon"></span>
              </button>

              <div
                className="collapse navbar-collapse"
                id="navbarSupportedContent"
              >
                <ul className="navbar-nav ml-auto">
                  <li className="nav-item active">
                    <NavLink
                      activeClassName="menu_active"
                      exact
                      className="nav-link"
                      to="/"
                    >
                      Home 
                      <span className="sr-only">(current)</span>
                    </NavLink>
                  </li>
                  {/* <li className="nav-item">
                  <NavLink
                      activeClassName="menu_active"
                      className="nav-link"
                      to="/Service "
                    >
                      Service
                    </NavLink>
                  </li> */}

                  <li className="nav-item">
                    <NavLink
                      activeClassName="menu_active"
                      className="nav-link"
                      to="/services"
                    >
                      Service
                    </NavLink>
                  </li>

                  <li className="nav-item">
                    <NavLink
                      activeClassName="menu_active"
                      className="nav-link"
                      to="/contact"
                    >
                      Contact
                    </NavLink>
                  </li>
                  {!userDetails ?<li className="nav-item">
                    <NavLink
                      activeClassName="menu_active"
                      className="nav-link"
                      to="/login"
                    >
                      Login
                    </NavLink>
                    </li> : ""}
                    {userDetails ? <li className="nav-item">
                        <NavLink
                          activeClassName="menu_active"
                          className="nav-link"
                          to="/ai"
                        >
                          Dashboard
                        </NavLink>

                        </li> : ""}
                  {userDetails ?<li className="nav-item">
                        <NavLink
                          activeClassName="menu_active"
                          className="nav-link"
                          onClick={logoutHandler}
                          to="/"
                        >
                          Logout
                        </NavLink>
                        </li> : ""}
                </ul>
              </div>
            </nav>
          </div>
        </div>
      </div>
    </>
  );
};

export default Navbar;
